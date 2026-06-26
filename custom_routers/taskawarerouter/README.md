# TaskAwareRouter

TaskAwareRouter is a custom inference router for LLMRouter that selects an appropriate LLM based on task domain and estimated complexity that selects the optimal LLM based on
two signals extracted from the user query:

1. **Task type** — coding / design / planning / research
2. **Complexity** — simple / complex

## How it works

A small cheap LLM (qwen2.5-7b) judges the incoming query and returns
task type and complexity as JSON. The router then looks up the right
model from a task map built from available llm_data.

If the judge LLM fails (timeout, missing API key, network error),
the router falls back to keyword matching and safe defaults.
If task classification fails, the router falls back to rule-based keyword routing to ensure deterministic behavior.

## PIPELINE
User Query
      │
      ▼
Judge LLM (Qwen2.5-7B)
      │
      ▼
Task + Complexity
      │
      ▼
Routing Table
      │
      ▼
Selected LLM


## Routing table

| Task     | Simple                      | Complex                          |
|----------|-----------------------------|----------------------------------|
| coding   | mistral-7b-instruct-v0.3    | llama3-70b-instruct              |
| design   | llama-3.1-8b-instruct       | mixtral-8x22b-instruct-v0.1      |
| planning | qwen2.5-7b-instruct         | llama-3.3-nemotron-super-49b-v1  |
| research | qwen2.5-7b-instruct         | mixtral-8x22b-instruct-v0.1      |
| language | llama-3.1-8b-instruct       | mixtral-8x22b-instruct-v0.1      |

## Free tier limitation

During development and validation, only `meta/llama-3.1-8b-instruct`
was available under the free NVIDIA NIM tier. All task_map entries
currently point to this model. The routing logic correctly selects
model tiers — swap in a full-access API key to activate
differentiated routing across all models.



## Usage

```bash
llmrouter infer \
  --router taskawarerouter \
  --config custom_routers/taskawarerouter/config.yaml \
  --query "Build a production grade authentication system" \
  --route-only
```

## Motivation

No existing router in LLMRouter routes by task domain and complexity
combined. This router fills that gap using a lightweight LLM judge
instead of hardcoded keywords or training data.



## Running tests

```bash
python -m pytest custom_routers/taskawarerouter/test_router.py -v
```

## Limitations

- Task map model names must exist in your llm_data config
- Judge LLM requires a valid API_KEYS environment variable
- Task classification accuracy depends on judge model quality
- Free NVIDIA tier limits available models for validation

## Why this router

No existing router in LLMRouter routes by task domain and complexity
combined. This router fills that gap using a lightweight LLM judge
instead of hardcoded keywords or training data. It handles any human
language naturally without requiring predefined keyword lists.

## Future Work

- Confidence-based routing
- Cost-aware model selection
- Latency-aware routing
- User feedback learning

## Author
Vidhursh Kumar V
GitHub: @Vidhursh-16