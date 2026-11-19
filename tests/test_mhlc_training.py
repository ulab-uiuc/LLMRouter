"""CPU regressions for MHLC completion masking and failure-aware training."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from llmrouter.models.gnosis_latent_router.backend.aux_head_shared_utils import (
    ChatBatchBuilder,
)
from llmrouter.models.gnosis_latent_router.backend.train_head_standalone_vllm_regression_weighted_multimodel import (
    _weighted_regression_terms,
)


def test_mask_keeps_assistant_content_and_supports_legacy_heads():
    # Qwen wraps answers with a role header, a possibly empty reasoning block,
    # and a closing marker. Use real tokenization with a small local vocabulary.
    words = [
        "[UNK]",
        "[PAD]",
        "user",
        "assistant",
        "<think>",
        "</think>",
        "end",
        "question",
        "longer",
        "reason",
        "answer",
    ]
    backend = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        padding_side="left",
        additional_special_tokens=["<think>", "</think>"],
        chat_template=(
            "{% for m in messages %}{{ m.role + ' ' }}"
            "{% if m.role == 'assistant' %}{{ '<think> ' + m.get('reasoning_content', '') + ' </think> ' }}{% endif %}"
            "{{ m.content + ' end ' }}{% endfor %}"
        ),
    )
    chats = [
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        [
            {"role": "user", "content": "longer question"},
            {"role": "assistant", "reasoning_content": "reason", "content": "answer"},
        ],
    ]

    def score_tokens(version):
        builder = ChatBatchBuilder(
            tokenizer, 128, "completion_text_only", feature_backend_version=version
        )
        batch = builder.build_from_messages(chats, [None, None])
        return [
            tokenizer.convert_ids_to_tokens(ids[mask.bool()].tolist())
            for ids, mask in zip(batch["input_ids"], batch["head_token_mask"])
        ]

    assert score_tokens(2) == [["answer"], ["reason", "answer"]]
    assert score_tokens(1)[0] == ["assistant", "answer", "end"]

    full = ChatBatchBuilder(tokenizer, 128, "completion_text_only").build_from_messages(
        chats[:1], [None]
    )
    limit = int(full["attention_mask"].sum()) - 1
    truncated = ChatBatchBuilder(
        tokenizer, limit + 1, "completion_text_only"
    ).build_from_messages(chats[:1], [None])
    assert truncated["gnosis_prompt_texts"] == [full["input_ids"][0, :limit].tolist()]
    assert truncated["attention_mask"].sum() == limit
    assert (
        tokenizer.decode(
            truncated["input_ids"][0, truncated["head_token_mask"][0].bool()]
        )
        == "answer"
    )


@pytest.mark.parametrize("loss_name", ["mse", "smooth_l1", "bce"])
def test_weighted_accumulation_agrees_with_full_batch(loss_name):
    scores = torch.tensor([0.25, 0.75, 0.90], requires_grad=True)
    labels = torch.tensor([0.0, 0.5, 1.0])
    weighted_sum, weight = _weighted_regression_terms(
        scores, labels, loss_name, 0.5, 3, 1, 1
    )
    (full_gradient,) = torch.autograd.grad(weighted_sum / weight, scores)
    total_weight = 0
    for score, label in zip(scores.split(1), labels.split(1)):
        numerator, denominator = _weighted_regression_terms(
            score, label, loss_name, 0.5, 3, 1, 1
        )
        numerator.backward()
        total_weight += denominator.item()
    torch.testing.assert_close(scores.grad / total_weight, full_gradient)
    # Default batch size one still upweights failures relative to successes.
    assert abs(full_gradient[0]) > abs(full_gradient[1])
