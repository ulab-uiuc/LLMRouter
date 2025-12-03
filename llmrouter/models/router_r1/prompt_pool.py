PROMPT_TEMPLATE_QWEN = """
Answer the given question. \
Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. \
After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a query inside <search> LLM-Name:Your-Query </search>. \

!!! STRICT FORMAT RULES for <search>: !!!
    + You MUST replace LLM-Name with the EXACT name of a model selected from [Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, LLaMA-3.1-70B-Instruct, Mistral-7B-Instruct, Mixtral-8x22B-Instruct, Gemma-2-27B-Instruct]. \
    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <search>.
    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly. \

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about: \
    + Why external information is needed. \
    + Which model is best suited for answering it, based on the LLMs' abilities (described below). \

When you call an LLM, the response will be returned between <information> and </information>. \
You must not limit yourself to repeatedly calling a single LLM (unless its provided information is consistently the most effective and informative). \
You are encouraged to explore and utilize different LLMs to better understand their respective strengths and weaknesses. \
It is also acceptable—and recommended—to call different LLMs multiple times for the same input question to gather more comprehensive information. \


#### The Descriptions of Each LLM \

Qwen2.5-7B-Instruct:\
Qwen2.5-7B-Instruct is a powerful Chinese-English instruction-tuned large language model designed for tasks in language, \
coding, mathematics, and reasoning. As part of the Qwen2.5 series, it features enhanced knowledge, stronger coding and \
math abilities, improved instruction following, better handling of long and structured texts, and supports up to 128K \
context tokens. It also offers multilingual capabilities across over 29 languages.\


LLaMA-3.1-8B-Instruct:\
LLaMA-3.1-8B-Instruct is an 8-billion-parameter instruction-tuned language model optimized for multilingual dialogue. \
It provides strong language understanding, reasoning, and text generation performance, outperforming many open-source \
and closed-source models on standard industry benchmarks.\


LLaMA-3.1-70B-Instruct:\
LLaMA-3.1-70B-Instruct is a 70-billion-parameter state-of-the-art language model designed for advanced multilingual \
dialogue tasks. It excels in language comprehension, complex reasoning, and high-quality text generation, setting a new \
standard against both open and closed models in benchmark evaluations.\


Mistral-7B-Instruct:\
Mistral-7B-Instruct is a fine-tuned version of the Mistral-7B-v0.3 language model designed to follow instructions, \
complete user requests, and generate creative text. It was trained on diverse public conversation datasets to enhance \
its ability to handle interactive tasks effectively.\


Mixtral-8x22B-Instruct:\
Mixtral-8x22B-Instruct is a cutting-edge sparse Mixture-of-Experts (SMoE) large language model from MistralAI. It \
efficiently uses 39B active parameters out of 141B total, delivering high performance at lower costs. The model excels \
at following instructions, completing tasks, and generating creative text, with strong skills in multiple languages \
(English, French, Italian, German, Spanish), mathematics, and coding. It also supports native function calling and \
handles long contexts up to 64K tokens for better information recall.\


Gemma-2-27B-Instruct:\
Gemma-2-27B-Instruct is a cutting-edge, instruction-tuned text generation model developed by Google. Built using the \
same technology as Gemini, it excels at text understanding, transformation, and code generation. As a lightweight, \
decoder-only model with open weights, it is ideal for tasks like question answering, summarization, and reasoning. \
Its compact size enables deployment on laptops, desktops, or private cloud setups, making powerful AI more accessible.\


If you find that no further external knowledge is needed, you can directly provide your final answer inside <answer> ... </answer>, without additional explanation or illustration. \
For example: <answer> Beijing </answer>. \
    + Important: You must not output the placeholder text "<answer> and </answer>" alone. \
    + You must insert your actual answer between <answer> and </answer>, following the correct format. \
Question: {question}\n
"""


PROMPT_TEMPLATE_LLAMA = """
Answer the given question. \
Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. \
After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a query inside <search> LLM-Name:Your-Query </search>. \

!!! STRICT FORMAT RULES for <search>: !!!
    + You MUST replace LLM-Name with the EXACT name of a model selected from [Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, LLaMA-3.1-70B-Instruct, Mistral-7B-Instruct, Mixtral-8x22B-Instruct, Gemma-2-27B-Instruct]. \
    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <search>.
    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly. \
    + DO NOT output <think> ... </think> as a literal string. Instead, perform your reasoning and write your thought process within these tags. That means: put your reasoning inside the tags, not as visible raw tags in the output. Only the reasoning content should appear between <think> and </think>. Do not explain or comment on the use of tags. \

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about: \
    + Why external information is needed. \
    + Which model is best suited for answering it, based on the LLMs' abilities (described below). \

When you call an LLM, the response will be returned between <information> and </information>. \
You must not limit yourself to repeatedly calling a single LLM (unless its provided information is consistently the most effective and informative). \
You are encouraged to explore and utilize different LLMs to better understand their respective strengths and weaknesses. \
It is also acceptable—and recommended—to call different LLMs multiple times for the same input question to gather more comprehensive information. \


#### The Descriptions of Each LLM \

Qwen2.5-7B-Instruct:\
Qwen2.5-7B-Instruct is a powerful Chinese-English instruction-tuned large language model designed for tasks in language, \
coding, mathematics, and reasoning. As part of the Qwen2.5 series, it features enhanced knowledge, stronger coding and \
math abilities, improved instruction following, better handling of long and structured texts, and supports up to 128K \
context tokens. It also offers multilingual capabilities across over 29 languages.\


LLaMA-3.1-8B-Instruct:\
LLaMA-3.1-8B-Instruct is an 8-billion-parameter instruction-tuned language model optimized for multilingual dialogue. \
It provides strong language understanding, reasoning, and text generation performance, outperforming many open-source \
and closed-source models on standard industry benchmarks.\


LLaMA-3.1-70B-Instruct:\
LLaMA-3.1-70B-Instruct is a 70-billion-parameter state-of-the-art language model designed for advanced multilingual \
dialogue tasks. It excels in language comprehension, complex reasoning, and high-quality text generation, setting a new \
standard against both open and closed models in benchmark evaluations.\


Mistral-7B-Instruct:\
Mistral-7B-Instruct is a fine-tuned version of the Mistral-7B-v0.3 language model designed to follow instructions, \
complete user requests, and generate creative text. It was trained on diverse public conversation datasets to enhance \
its ability to handle interactive tasks effectively.\


Mixtral-8x22B-Instruct:\
Mixtral-8x22B-Instruct is a cutting-edge sparse Mixture-of-Experts (SMoE) large language model from MistralAI. It \
efficiently uses 39B active parameters out of 141B total, delivering high performance at lower costs. The model excels \
at following instructions, completing tasks, and generating creative text, with strong skills in multiple languages \
(English, French, Italian, German, Spanish), mathematics, and coding. It also supports native function calling and \
handles long contexts up to 64K tokens for better information recall.\


Gemma-2-27B-Instruct:\
Gemma-2-27B-Instruct is a cutting-edge, instruction-tuned text generation model developed by Google. Built using the \
same technology as Gemini, it excels at text understanding, transformation, and code generation. As a lightweight, \
decoder-only model with open weights, it is ideal for tasks like question answering, summarization, and reasoning. \
Its compact size enables deployment on laptops, desktops, or private cloud setups, making powerful AI more accessible.\


If you find that no further external knowledge is needed, you can directly provide your final answer inside <answer> ... </answer>, without additional explanation or illustration. \
For example: <answer> Beijing </answer>. \
    + Important: You must not output the placeholder text "<answer> and </answer>" alone. \
    + You must insert your actual answer between <answer> and </answer>, following the correct format. \
Question: {question}\n
"""
