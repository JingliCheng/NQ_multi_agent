# Multi-Agent-Autogen.py

import os
import openai
from typing import Dict, Tuple
import json
import time
import re
from ratelimit import limits, sleep_and_retry, RateLimitException
from concurrent.futures import ThreadPoolExecutor
from abc import ABC
from autogen import ConversableAgent
# Import BaseAgentSystem from Original Repo
from nq_agents import multi_agent


# QPS and concurrency limits
MAX_QPS = 2  # 每秒最多请求数
MAX_CONCURRENT_REQUESTS = 2  # 最大并发请求数

# Token limit for different models
DEFAULT_TOKEN_LIMIT = 8192  # 默认 Token 限制（8k）
LLAMA_TOKEN_LIMIT = 20480   # Llama 模型 Token 限制（80k）#todo: 80k? 20480?
OPENAI_TOKEN_LIMIT = 20480 # OpenAI GPT-4o Turbo 模型 Token 限制（100k）


# Load data from file
DEV_FILE_10 = 'data/v1.0-simplified_nq-dev-all_sample10_seed44.jsonl'
DEV_FILE_100 = 'data/v1.0-simplified_nq-dev-all_sample100_seed42.jsonl'
SINGLE_ENTRY_FILE= 'data/first_entry_sample.jsonl'

# Configure LLM provider
LLM_PROVIDER = "ollama"
OLLAMA_API_BASE = "http://localhost:11434/v1"
MAX_TOKENS = LLAMA_TOKEN_LIMIT if LLM_PROVIDER == "ollama" else OPENAI_TOKEN_LIMIT

# def get_nq_tokens(simplified_nq_example):
#   """Returns list of blank separated tokens."""

#   if "document_text" not in simplified_nq_example:
#     raise ValueError("`get_nq_tokens` should be called on a simplified NQ"
#                      "example that contains the `document_text` field.")

#   return simplified_nq_example["document_text"].split(" ")

# def get_short_answers(nq_example):
#     document_tokens = get_nq_tokens(nq_example)
#     print(len(nq_example['annotations']))
#     short_answers = []
#     for annotation in nq_example['annotations']:
#         if annotation['short_answers']:
#             for short_answer in annotation['short_answers']:
#                 short_answer_text = ' '.join(
#                     document_tokens[
#                         short_answer['start_token']:short_answer['end_token']]
#                     )
#                 short_answers.append(short_answer_text)
#     return short_answers


class MultiAgentAutogen(multi_agent.BaseAgentSystem):
    def __init__(
        self,
        llm_provider="ollama",
        api_key=None,
        max_tokens=20480,
        max_qps=2,
        max_concurrent_requests=2,
    ):
        """
        Initialize the multi-agent autogen system.

        Args:
            llm_provider (str): The LLM provider, e.g., 'openai' or 'ollama'.
            api_key (str): API key for the LLM provider.
            max_tokens (int): Maximum tokens for the LLM.
            max_qps (int): Maximum queries per second.
            max_concurrent_requests (int): Maximum concurrent requests.
        """
        super().__init__()
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.max_qps = max_qps
        self.max_concurrent_requests = max_concurrent_requests

        # Set up LLM configuration
        self.llm_config = self.get_llm_config()

    def get_llm_config(self):
        # todo: do we need to set the temperature to 0.7? Should we set it to 0?
        llm_configs = {
            "ollama": {
                "config_list": [
                    {
                        "model": "llama3.2:latest",
                        "api_key": "ollama",
                        "base_url": "http://localhost:11434/v1",
                        "temperature": 0.0,
                    }
                ]
            },
            "openai": {
                "config_list": [
                    {
                        "model": "gpt-4o",
                        "api_key": self.api_key,
                        "temperature": 0.0,
                    }
                ]
            },
        }
        if self.llm_provider not in llm_configs:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        return llm_configs[self.llm_provider]

    def create_extract_content_agent(self):
        return ConversableAgent(
            name="ExtractContentAgent",
            system_message="""
#Role:
You are an expert text extractor that identifies the exact passage from a given context that directly answers the specific question.

# Instructions

1. **Extract Exact Text**: Provide the **exact subsequences** from the context that may **contain or directly answer the question**. The extracted text must be **word-for-word from the original context** without any changes.
2. **No Alterations**: Do not paraphrase, summarize, or add any additional information. Avoid introducing personal opinions or external knowledge.
3. **Multiple Passages**: If multiple parts of the context are relevant, include all relevant passages.
4. **Step-by-Step Analysis**: Carefully analyze the context step-by-step to identify all passages that are relevant to the question.
5. **Formatting**: Your output must follow the following format.
6. **Word index** Ignore word index when analyze and extract the content. Output a copy of extracted content with index.
7. **Reasoning** Provide a reasoning process for your extraction. Your reasoning should be short and concise.
8. **Empty Output** If you cannot find any relevant context, return an empty output.

# Format
## Input Format

[Question] Question Text
[Context] Document Text

## Output Format
{
    "reasoning": "Reasoning process",
    "index_context": "Relevant Context with word index",
    "relevant_context": "Relevant Context without word index"
}

# Examples
## Example 1
**Input:**
[Question] What is the capital city of Japan?
[Context] [wd_idx[0]]Tokyo is the capital city of Japan. It is one [wd_idx[10]]of the largest cities in the world.

**Output:**
{
    "reasoning": "This passage contains the capital city of Japan, which is Tokyo.",
    "index_context": "[wd_idx[0]]Tokyo is the capital city of Japan.",
    "Relevant_Context": "Tokyo is the capital city of Japan."
}

## Example 2
**Input:**
[Question] Which element has the atomic number 6?
[Context] [wd_idx[290]]Carbon has the atomic number 6 and is essential to [wd_idx[300]]all known life forms.

**Output:**
{
    "reasoning": "This passage contains the element with atomic number 6, which is Carbon.",
    "index_context": "[wd_idx[290]]Carbon has the atomic number 6 and is essential to [wd_idx[300]]all known life forms.",
    "Relevant_Context": "Carbon has the atomic number 6 and is essential to all known life forms."
}

## Example 3 Empty Output
**Input:**
[Question] Who is the author of the book "To Kill a Mockingbird"?
[Context] Many people [wd_idx[140]]know the book "To Kill a Mockingbird" from the [wd_idx[150]]movie, directed by Robert Mulligan.

**Output:**
{
    "reasoning": "This passage only mentions the movie and its director, not the author of the book.",
    "index_context": null,
    "Relevant_Context": null,
}
""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )

    def create_judger_agent(self):
        return ConversableAgent(
            name="JudgerAgent",
            system_message="""
You are a judging agent who evaluates the relevance of extracted passages to the given question.

Your task is to:
1. Assess each extracted passage and rate its relevance to the question.
2. Choose the top 3 most relevant passages based on their relevance scores.
3. Return the top passages in the **exact JSON format** as shown below. Do not include any additional explanations or ratings.

# Output Format
{
    "messages": ["Passage 1", "Passage 2", "Passage 3"]
}

# Examples
## Example 1
Input:
Question: What is the capital of France?
Passages: ["Paris is the capital of France.", "France has many beautiful cities, including Lyon and Nice.", "The Eiffel Tower is located in Paris.", "Marseille is an important port city in France."]

Output:
{
    "messages": ["Paris is the capital of France.", "The Eiffel Tower is located in Paris."]
}

## Example 2
Input:
Question: Who painted the Mona Lisa?
Passages: ["Leonardo da Vinci painted the Mona Lisa.", "Vincent van Gogh was known for Starry Night.", "The Mona Lisa is a famous painting."]

Output:
{
    "messages": ["Leonardo da Vinci painted the Mona Lisa."]
}
    """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
    
    def create_cut_answer_agent(self):
        return ConversableAgent(
            name="CutAnswerAgent",
            system_message="""
You are a professional judge on the question and answer pair. Your task is to cut the answer to the shortest possible substring that can still answer the question correctly.

Instructions:
1. Analyze the given answer carefully.
2. Extract the shortest contiguous substring that directly answers the question.
3. Return the output in the **exact JSON format** as shown below.

# Output Format
{
    "messages": ["The shortest substring answer"]
}

# Examples
## Example 1
Input:
Question: What is the capital of Japan?
Initial answer: The capital of Japan is Tokyo, which is known for its culture.

Output:
{
    "messages": ["Tokyo"]
}

## Example 2
Input:
Question: Who wrote the play Hamlet?
Initial answer: The play Hamlet was written by William Shakespeare.

Output:
{
    "messages": ["William Shakespeare"]
}
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
    
    def create_refine_answer_agent(self):
        return ConversableAgent(
            name="RefineAnswerAgent",
            system_message="""
You are a professional judge on the question and answer pair. Your task is to refine the answer to the shortest possible substring without losing the meaning or correctness.

Instructions:
1. Analyze the provided answer and refine it.
2. Return the most concise and correct substring that answers the question.
3. Return the output in the **exact JSON format** as shown below.

# Output Format
{
    "messages": ["The final refined answer"]
}

# Examples
## Example 1
Input:
Question: Who painted the Mona Lisa?
Previous answer: The Mona Lisa was painted by Leonardo da Vinci.

Output:
{
    "messages": ["Leonardo da Vinci"]
}

## Example 2
Input:
Question: What is the tallest mountain in the world?
Previous answer: Mount Everest is the tallest mountain in the world.

Output:
{
    "messages": ["Mount Everest"]
}
""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )

    def get_tokenizer(self, model_name="gpt-4o"):
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception as e:
            print(f"Error loading tokenizer for model {model_name}: {e}")
            return tiktoken.get_encoding("gpt-4o")

    def split_document(self, document, model_name="gpt-4o", max_tokens=8192, overlap=0):
        tokenizer = self.get_tokenizer(model_name)
        tokens = tokenizer.encode(document)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap
        return chunks

    def extract_content(self, agent, chunk, question):
        input_message = f"[Question] {question}\n[Context] {chunk}"
        response = agent.generate_reply(
            messages=[{"content": input_message, "role": "user"}]
        )
        try:
            response_content = response["messages"][-1]["content"]
            response_json = json.loads(response_content)
            relevant_context = response_json.get("Relevant_Context", "")
            return relevant_context
        except Exception as e:
            return ""

    def judge_relevance(self, agent, extracted_contents, question):
        input_message = (
            f"Question: {question}\n"
            f"Passages: {extracted_contents}"
        )
        response = agent.generate_reply(
            messages=[{"content": input_message, "role": "user"}]
        )
        try:
            # print("response:", response)
            # Convert to json if response is a string
            if isinstance(response, str):
                try:
                    response_json = json.loads(response)
                    # convert successfully and check if it contains "messages"
                    if isinstance(response_json, dict) and "messages" in response_json:
                        messages = response_json["messages"]
                        if isinstance(messages, list):
                            return messages
                except json.JSONDecodeError:
                    # If response is not valid JSON, assume plain text format
                    print("Warning: Response is not valid JSON. Assuming plain text format.")

            # If response is not a valid JSON, return it as a list
            if isinstance(response, str):
                return [response.strip()]
        
        except json.JSONDecodeError:
            print("Error: Response is not a valid JSON. Returning empty list.")
            print("Received response:", response)
            return []
        except Exception as e:
            print(f"Unexpected error in judge_relevance: {e}")
            return []
        
    def cut_answer(self, cut_agent, question: str, initial_answer: str) -> str:
        input_message = f"Question: {question}\nInitial answer: {initial_answer}"
        response = cut_agent.generate_reply(
            messages=[{"content": input_message, "role": "user"}]
        )

        try:
            # 检查 response 是否为字符串格式
            if isinstance(response, str):
                return response.strip()

            # 检查 response 是否为包含 "messages" 键的字典
            if isinstance(response, dict) and "messages" in response:
                messages = response["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    return messages[0].strip()

            # 如果都不符合，返回初始答案
            print("Warning: Unrecognized response format. Returning initial answer.")
            return initial_answer
        except Exception as e:
            print(f"Unexpected error in _cut_answer: {e}")
            return initial_answer

    def refine_answer(self, refine_agent, question: str, previous_answer: str) -> str:
        input_message = f"Question: {question}\nPrevious answer: {previous_answer}"
        response = refine_agent.generate_reply(
            messages=[{"content": input_message, "role": "user"}]
        )

        try:
            # 检查 response 是否为字符串格式
            if isinstance(response, str):
                # return response.strip()
                return response

            # 检查 response 是否为包含 "messages" 键的字典
            if isinstance(response, dict) and "messages" in response:
                messages = response["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    # return messages[0].strip()
                    return messages[0]

            # 如果都不符合，返回之前的答案
            print("Warning: Unrecognized response format. Returning previous answer.")
            return previous_answer
        except Exception as e:
            print(f"Unexpected error in _refine_answer: {e}")
            return previous_answer

    def predict(self, example: Dict, verbose: bool = False) -> Tuple[str, float]:
        """
        Make prediction using the multi-agent system.

        Args:
            example (Dict): Single Natural Questions example.
            verbose (bool): Whether to print intermediate steps.

        Returns:
            Tuple[str, float]: Predicted answer and its score.
        """
        document = example["document_text"]
        question = example["question_text"]
        print(f"question: {question}")
        # print(f"example: {example}")
        # Step 1: Split the document into chunks
        chunks = self.split_document(
            document, model_name="gpt-4o", max_tokens=self.max_tokens, overlap=100
        )

        # Step 2: Create ExtractContentAgent for each chunk
        extract_agents = [self.create_extract_content_agent() for _ in chunks]

        # Step 3: Extract content from each chunk
        @sleep_and_retry
        @limits(calls=self.max_qps, period=1)
        def extract_content_with_rate_limit(agent, chunk, question):
            while True:
                try:
                    return self.extract_content(agent, chunk, question)
                except RateLimitException:
                    time.sleep(1)


        extracted_contents = []

        with ThreadPoolExecutor(
            max_workers=self.max_concurrent_requests
        ) as executor:
            # print("executor: ", executor)
            future_to_chunk = {
                executor.submit(
                    extract_content_with_rate_limit, agent, chunk, question
                ): chunk
                for agent, chunk in zip(extract_agents, chunks)
            }
            # print("future_to_chunk: ", future_to_chunk)
            for future in future_to_chunk:
                try:
                    # todo: wrong. However, there is an issue here: it appears that the result is assigned as future_to_chunk[future], which is the chunk rather than the result of the future itself. To get the actual result, it should instead be:
                    result = future.result()
                    extracted_contents.append(result)
                    print(f"result: {result}")
                except Exception as e:
                    if verbose:
                        print(f"Error during extraction: {e}")

        # for future, chunk in future_to_chunk.items():
            # print(f"Future: {future}")
            # print(f"Chunk (first 200 characters): {chunk[:200]}...\n")

        # Step 4: Use JudgerAgent to judge relevance
        judger_agent = self.create_judger_agent()
        messages = self.judge_relevance(judger_agent, extracted_contents, question)

        if not messages:
            print("No relevant passages found.")
            return "", 0.0

        initial_answer = messages[0]
        if verbose:
            print(f"Initial answer: {initial_answer}")
        
        # Step 5: Use CutAnswerAgent to cut the answer
        cut_agent = self.create_cut_answer_agent()
        cut_answer = self.cut_answer(cut_agent, question, initial_answer)
        if verbose:
            print(f"Cut answer: {cut_answer}")

        # Step 6: Use RefineAnswerAgent to refine the answer
        refine_agent = self.create_refine_answer_agent()
        refined_answer = self.refine_answer(refine_agent, question, cut_answer)
        if verbose:
            print(f"Refined answer: {refined_answer}")

        # Placeholder score
        score = 50.0

        if verbose:
            print(f"Question: {question}")
            print(f'gold answer: {example["annotations"][0]["short_answers"]}')
            print(f"gold answer: {get_short_answers(example)}")
            print(f"Predicted Answer: {refined_answer}")

        # convert refined_answer to dict
            # 去掉首尾的空白字符（包括换行）
        try:
            # 去掉首尾的空白字符（包括换行）
            refined_answer = refined_answer.strip()

            # 使用正则表达式提取 JSON 对象部分
            json_match = re.search(r"\{.*\}", refined_answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # 解析 JSON
                refined_answer = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in the refined_answer.")

        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            print("Received refined_answer:", refined_answer)
            # 如果解析失败，返回原始字符串作为备用
            refined_answer = {"messages": [refined_answer]}

        except Exception as e:
            print(f"Unexpected error: {e}")
            refined_answer = {"messages": [refined_answer]}

        print(f"Parsed refined_answer: {refined_answer}")
        return refined_answer['messages'][0], score