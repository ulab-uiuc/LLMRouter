import time

import openai
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool


_cached_client = None


def get_client(
    base_url="",
    api_key="",
    max_retries=2,
    timeout=60
):
    global _cached_client
    if _cached_client is None:
        _cached_client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout
        )
    return _cached_client


def get_llm_response_via_api(prompt,
                             LLM_MODEL="",
                             base_url="",
                             api_key="",
                             TAU=1.0,
                             TOP_P=1.0,
                             SEED=42,
                             MAX_TRIALS=500,
                             TIME_GAP=5):
    '''
    res = get_llm_response_via_api(prompt='hello')  # Default: TAU Sampling (TAU=1.0)
    res = get_llm_response_via_api(prompt='hello', TAU=0)  # Greedy Decoding
    res = get_llm_response_via_api(prompt='hello', TAU=0.5, N=2, SEED=None)  # Return Multiple Responses w/ TAU Sampling
    '''
    client = get_client(base_url=base_url, api_key=api_key)
    completion = None
    while MAX_TRIALS:
        MAX_TRIALS -= 1
        try:
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=TAU,
                top_p=TOP_P,
                seed=SEED,
                max_tokens=512,
            )
            break
        except Exception as e:
            print(e)
            if "request timed out" in str(e).strip().lower():
                break
            print("Retrying...")
            time.sleep(TIME_GAP)

    if completion is None:
        raise Exception("Reach MAX_TRIALS={}".format(MAX_TRIALS))
    contents = completion.choices
    meta_info = completion.usage
    completion_tokens = meta_info.completion_tokens
    # prompt_tokens = meta_info.prompt_tokens
    # total_tokens = meta_info.total_tokens
    # print(completion_tokens, prompt_tokens, total_tokens)
    if len(contents) == 1:
        return contents[0].message.content, completion_tokens
    else:
        return [c.message.content for c in contents], completion_tokens


API_PRICE_1M_TOKENS = {
    "qwen/qwen2.5-7b-instruct": 0.3,
    "meta/llama-3.1-70b-instruct": 0.88,
    "meta/llama-3.1-8b-instruct": 0.18,
    "mistralai/mistral-7b-instruct-v0.3": 0.2,
    "mistralai/mixtral-8x22b-instruct-v0.1": 1.2,
    "google/gemma-2-27b-it": 0.8,
    "writer/palmyra-creative-122b": 1.8,
    "nvidia/llama3-chatqa-1.5-8b": 0.18,
}


AGENT_PROMPT = """
You are a helpful assistant. \
You are participating in a multi-agent reasoning process, where a base model delegates sub-questions to specialized models like you. \
\nYour task is to do your **absolute best** to either: \n
    + Answer the question directly, if possible, and provide a brief explanation; or \n
    + Offer helpful and relevant context, background knowledge, or insights related to the question, even if you cannot fully answer it. \

If you are completely unable to answer the question or provide any relevant or helpful information, you must: \n
    + Clearly state that you are unable to assist with this question, and \n
    + Explicitly instruct the base model to consult other LLMs for further assistance. \

**Important Constraints**: \n
    + Keep your response clear, concise, and informative (preferably under 512 tokens). Your response will help guide the base model’s reasoning and next steps. \n
    + Stay strictly on-topic. Do not include irrelevant or generic content. \

\n\nHere is the sub-question for you to assist with: {query}\n
"""


def request_task(data):
    q_id, query_text, TAU, LLM_NAME, api_base, api_key = data
    if LLM_NAME == "":
        print("LLM Name Error")
        return q_id, "LLM Name Error", 0.0
    print(LLM_NAME)
    try:
        input_prompt = AGENT_PROMPT.format_map({"query": query_text})
        single_response, completion_tokens = get_llm_response_via_api(prompt=input_prompt,
                                                                      base_url=api_base,
                                                                      api_key=api_key,
                                                                      TAU=TAU,
                                                                      LLM_MODEL=LLM_NAME)
        print(single_response, completion_tokens)
    except Exception as e:
        print(e)
        single_response = "API Request Error"
        completion_tokens = 0.0

    return q_id, single_response, int(completion_tokens) * API_PRICE_1M_TOKENS[LLM_NAME]


def check_llm_name(target_llm):
    TAU = 0
    LLM_NAME = ""
    if "qwen" in target_llm:
        LLM_NAME = "qwen/qwen2.5-7b-instruct"
    elif "palmyra" in target_llm or "creative" in target_llm:
        LLM_NAME = "writer/palmyra-creative-122b"
    elif "llama" in target_llm:
        if "70b" in target_llm:
            LLM_NAME = "meta/llama-3.1-70b-instruct"
        elif "51b" in target_llm:
            LLM_NAME = "nvidia/llama-3.1-nemotron-51b-instruct"
        elif "49b" in target_llm:
            LLM_NAME = "nvidia/llama-3.3-nemotron-super-49b-v1"
        elif "8b" in target_llm:
            if "chatqa" in target_llm:
                LLM_NAME = "nvidia/llama3-chatqa-1.5-8b"
            else:
                LLM_NAME = "meta/llama-3.1-8b-instruct"
        else:
            LLM_NAME = ""
    elif "mistral" in target_llm:
        LLM_NAME = "mistralai/mistral-7b-instruct-v0.3"
    elif "mixtral" in target_llm:
        LLM_NAME = "mistralai/mixtral-8x22b-instruct-v0.1"
    elif "granite" in target_llm:
        LLM_NAME = "ibm/granite-3.0-8b-instruct"
    elif "gemma" in target_llm:
        LLM_NAME = "google/gemma-2-27b-it"
        TAU = 0.1
    else:
        LLM_NAME = ""

    return LLM_NAME, TAU


def access_routing_pool(queries, api_base, api_key):
    task_args = []
    for q_id, single_query in enumerate(queries):
        target_llm = single_query.split(":")[0].strip().lower()
        query_text = single_query.split(":")[1]
        LLM_NAME, TAU = check_llm_name(target_llm=target_llm)
        task_args.append((q_id, query_text, TAU, LLM_NAME, api_base, api_key))

    ret = []
    with ThreadPool(10) as p:
        for r in tqdm(p.imap_unordered(request_task, task_args), total=len(task_args), desc="Processing", ncols=100):
            ret.append(r)

    ret.sort(key=lambda x: x[0], reverse=False)
    resp = []
    completion_tokens_list = []
    for _, response, completion_tokens in ret:
        resp.append(response)
        completion_tokens_list.append(completion_tokens)

    return {"result": resp, "completion_tokens_list": completion_tokens_list}
