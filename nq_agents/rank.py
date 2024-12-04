import re
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from autogen import ConversableAgent

LLM_PROVIDER = "ollama"
LLM_PROVIDERS = ["ollama", "openai"]

def get_llm_config(llm_provider=LLM_PROVIDER):
    """
    Returns the LLM configuration based on the selected provider.
    """
    llm_configs = {
        "ollama": {
            "config_list": [
                {
                    "model": "llama3.2:latest",
                    "api_key": "ollama",  # Replace with actual key if necessary
                    "base_url": "http://localhost:11434/v1",
                    "temperature": 0.7,  # Use 0.0 for deterministic results
                }
            ]
        },
        "openai": {
            "config_list": [
                {
                    "model": "gpt-4o",
                    "api_key": "your_openai_api_key",  # Replace with actual API key
                    "base_url": "https://api.openai.com/v1",
                    "temperature": 0.7,  # Set temperature for OpenAI
                }
            ]
        },
    }

    if llm_provider not in llm_configs:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    return llm_configs[llm_provider]

def create_rank_agent():
    llm_config = get_llm_config()
    return ConversableAgent(
        name="RankAgent",
        system_message="""
# Role
You are an assistant tasked with selecting the best candidate from a provided list based on a given question. You will be given no more than 5 candidates, please read it carefully and give me the ID of the best candidate.

# Task
1. Evaluate the relevance of each candidate's content to the question.
2. Choose the best candidate from the provided list of candidate IDs.
3. Your output must only be a single integer representing the ID of the best candidate.
4. Do not create new IDs, add explanations, or include any extra text—only return the selected ID.

# Input Format
Question: <question_text>
Candidates:
- ID: <candidate_id>, Content: <candidate_content>, Reasoning: <candidate_reasoning>

# Output Format
<integer>

# Example
**Input:**
Question: What is the capital of France?
Candidates:
        {"id": 1, "relevant_content": "Nice is a city in southern France.", "reasoning": "Not the capital."},
        {"id": 2, "relevant_content": "Berlin is the capital of Germany.", "reasoning": "Irrelevant to the question."},
        {"id": 3, "relevant_content": "Lyon is a city in France.", "reasoning": "Not the capital but relevant."},
        {"id": 4, "relevant_content": "Madrid is the capital of Spain.", "reasoning": "Not relevant to France."},
        {"id": 5, "relevant_content": "Marseille is a major city in France.", "reasoning": "Not the capital."},

**Output:**
1
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

def rank_candidates_with_agent(agent, question: str, candidates: List[Dict], max_retries: int = 3) -> int:
    """
    Use the RankAgent to select the best candidate, with retries for invalid outputs.

    :param agent: The RankAgent instance.
    :param question: The question to evaluate candidates against.
    :param candidates: List of candidate dictionaries with 'id', 'relevant_content', and 'reasoning'.
    :param max_retries: Maximum number of retries for invalid outputs.
    :return: The ID of the selected top candidate as an integer, or None if retries fail.
    """
    input_data = {
        "question": question,
        "candidates": candidates
    }

    input_message = json.dumps(input_data)
    messages = [{"role": "user", "content": input_message}]

    for attempt in range(max_retries):
        response = agent.generate_reply(messages)

        # 打印原始响应以调试问题
        print(f"Attempt {attempt + 1}/{max_retries}, Raw response from LLM: {response}")

        try:
            # 提取整数 ID
            int_match = re.search(r"\d+", response)
            if int_match:
                top_candidate_id = int(int_match.group(0))

                # 验证 ID 是否在候选列表中
                if any(candidate["id"] == top_candidate_id for candidate in candidates):
                    return top_candidate_id
                else:
                    print(f"Invalid ID {top_candidate_id} not in batch. Retrying ({attempt + 1}/{max_retries})...")
            else:
                print(f"No valid integer ID found in response. Retrying ({attempt + 1}/{max_retries})...")
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Retrying ({attempt + 1}/{max_retries})...")

    # 如果尝试次数用尽，返回 None 并打印最终无效响应
    print(f"Failed to get a valid ID after {max_retries} attempts. Final response: {response}")
    return None
def process_batch(rank_agent, question, batch):
    """
    Process a single batch of candidates to select the top one.
    """
    try:
        top_candidate_id = rank_candidates_with_agent(rank_agent, question, batch)
        print(f"Top candidate ID: {top_candidate_id}")
        print(f"Candidate IDs in batch: {[candidate['id'] for candidate in batch]}")

        # 确保 top_candidate_id 在 batch 中存在
        if any(candidate["id"] == top_candidate_id for candidate in batch):
            return next(candidate for candidate in batch if candidate["id"] == top_candidate_id)
        else:
            print(f"Warning: Invalid ID {top_candidate_id} returned by LLM. Skipping this batch.")
            return None  # 返回 None 表示跳过该批次
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None
    
def rank(question: str, candidates: List[Dict], batch_size: int = 5) -> int:
    rank_agent = create_rank_agent()
    round_number = 1

    while len(candidates) > 1:
        print(f"Round {round_number}: {len(candidates)} candidates remaining.")
        next_round_candidates = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_batch, rank_agent, question, candidates[i:i + batch_size])
                for i in range(0, len(candidates), batch_size)
            ]
            for future in futures:
                result = future.result()
                if result:  # 过滤掉 None
                    next_round_candidates.append(result)

        if not next_round_candidates:
            print("Error: All batches were skipped due to invalid LLM outputs.")
            return None  # 或者抛出异常

        candidates = next_round_candidates
        round_number += 1

    return candidates[0]["id"] if candidates else None


if __name__ == "__main__":
    question = "What is the capital of France?"
    candidates = [
        {"id": 1, "relevant_content": "Nice is a city in southern France.", "reasoning": "Not the capital."},
        {"id": 2, "relevant_content": "Berlin is the capital of Germany.", "reasoning": "Irrelevant to the question."},
        {"id": 3, "relevant_content": "Lyon is a city in France.", "reasoning": "Not the capital but relevant."},
        {"id": 4, "relevant_content": "Madrid is the capital of Spain.", "reasoning": "Not relevant to France."},
        {"id": 5, "relevant_content": "Marseille is a major city in France.", "reasoning": "Not the capital."},
        {"id": 6, "relevant_content": "Paris is home to the Eiffel Tower.", "reasoning": "Relevant but not directly answering."},
        {"id": 7, "relevant_content": "Rome is the capital of Italy.", "reasoning": "Irrelevant to the question."},
        {"id": 8, "relevant_content": "Nice is a city in southern France.", "reasoning": "Not the capital."},
        {"id": 19, "relevant_content": "Paris has been the capital since the Middle Ages.", "reasoning": "Accurate and directly relevant."},
        {"id": 10, "relevant_content": "Paris is the political hub of France.", "reasoning": "Relevant and concise."},
        {"id": 11, "relevant_content": "Paris is the capital of France.", "reasoning": "Accurate and concise."},
        {"id": 12, "relevant_content": "Paris has been the capital since the Middle Ages.", "reasoning": "Accurate and directly relevant."},
        {"id": 13, "relevant_content": "Paris is the political hub of France.", "reasoning": "Relevant and concise."},
    ]

    best_candidate_id = rank(question, candidates)

    print(f"The top-ranked candidate ID: {best_candidate_id}")
