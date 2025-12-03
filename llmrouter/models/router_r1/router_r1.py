import re

import torch
import torch.nn as nn
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from llmrouter.models.router_r1.prompt_pool import *
from llmrouter.models.meta_router import MetaRouter
from llmrouter.models.router_r1.route_service import access_routing_pool



class RouterR1(MetaRouter):
    """
    Router-R1
    -----------
    Example router that performs R1-like routing.

    This class:
        - Inherits MetaRouter to reuse configuration and utilities
        - Implements the `route()` method using the underlying `model`
    """

    def __init__(self, model: nn.Module, yaml_path: str | None = None, resources=None):
        """
        Args:
            model (nn.Module):
                Underlying LLM  (e.g., qwen, llama, etc).
            yaml_path (str | None):
                Optional path to YAML config for this router.
            resources (Any, optional):
                Additional shared resources, if needed.
        """
        super().__init__(model=model, yaml_path=yaml_path, resources=resources)

    def route_single(self, query: Dict[str, Any], model_id: str, api_base: str, api_key: str):
        """
        Perform inference on Router-R1.
        """
        # Prepare the question
        question = query["query"].strip()
        if question[-1] != '?':
            question += '?'

        # Model path and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        llm = LLM(model=model_id, dtype="float16", tensor_parallel_size=torch.cuda.device_count())

        curr_route_template = '\n{output_text}\n<information>{route_results}</information>\n'

        # Initial prompt
        if model_id.lower().find("qwen") != -1:
            prompt = PROMPT_TEMPLATE_QWEN.format_map({"question": question})
        else:
            prompt = PROMPT_TEMPLATE_LLAMA.format_map({"question": question})
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                                tokenize=False)

        # Sampling configuration
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1024,
            stop=["</search>", "</answer>"]
        )

        cnt = 0
        print('\n\n################# [Start Reasoning + Routing] ##################\n\n')
        STOP = False
        all_output = ""

        while True:
            if cnt > 4:
                break
            outputs = llm.generate(prompt, sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text
            if output_text.find("<answer>") != -1:
                STOP = True
                output_text += "</answer>"
            if not STOP:
                output_text += "</search>"

            print(f"[Generation {cnt}] Output:\n{output_text}")

            tmp_query = self.get_query(output_text)
            if tmp_query:
                route_results = self.route(tmp_query, api_base=api_base, api_key=api_key)
            else:
                route_results = ''

            if not STOP:
                prompt += curr_route_template.format(output_text=output_text, route_results=route_results)
                all_output += curr_route_template.format(output_text=output_text, route_results=route_results)
            else:
                all_output += output_text + "\n"
                break

            cnt += 1

        print('\n\n################# [Output] ##################\n\n')

        print(all_output)

        print('\n\n################# [Output] ##################\n\n')

    def get_query(text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None

    def route(query, api_base, api_key):
        ret = access_routing_pool(
            queries=[query],
            api_base=api_base,
            api_key=api_key
        )
        return ret['result'][0]
