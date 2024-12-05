import re
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from autogen import ConversableAgent
import time

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
You are an assistant tasked with selecting the best candidate from a provided list based on a given question. Each candidate has an ID, content, and reasoning. 

# Task
1. Read the provided candidates carefully including their id.
2. Evaluate the relevance of each candidate's reasoning and content to the question.
3. Choose the best candidate by selecting the ID from the provided list. Do not use your own knowledge—only rely on the given candidates and their reasoning.
4. Your output must be a single integer representing the ID of the best candidate.
5. Do not create new IDs, provide explanations, or include extra text—only return the selected ID.

# Input Format
Question: <question_text>
Candidates:
- ID: <candidate_id>, Content: <candidate_content>, Reasoning: <candidate_reasoning>

# Output Format
<integer>

# Examples
**Input:**
Question: What is the capital of France?
Candidates:
- ID: 1, Content: "Paris is the capital of France.", Reasoning: "Accurate and concise."
- ID: 2, Content: "Berlin is the capital of Germany.", Reasoning: "Irrelevant to the question."
- ID: 3, Content: "Lyon is a city in France.", Reasoning: "Not the capital but relevant."
- ID: 4, Content: "Madrid is the capital of Spain.", Reasoning: "Not relevant to France."
- ID: 5, Content: "Marseille is a major city in France.", Reasoning: "Not the capital."

**Output:**
1

**Input:**
Question: What is the largest planet in our solar system?
Candidates:
- ID: 1, Content: "Mars is the largest planet in our solar system.", Reasoning: "Incorrect, Mars is not the largest planet."
- ID: 2, Content: "Earth is the third planet from the Sun.", Reasoning: "Not relevant to the question."
- ID: 3, Content: "Jupiter is the largest planet in our solar system.", Reasoning: "Correct and concise."
- ID: 4, Content: "Venus is the brightest planet.", Reasoning: "Not relevant to the question."
- ID: 5, Content: "Saturn is the second-largest planet.", Reasoning: "Not the largest."

**Output:**
3

**Input:**
Question: What is the chemical formula for water?
Candidates:
- ID: 1, Content: "H2O is the chemical formula for water.", Reasoning: "Correct and relevant."
- ID: 2, Content: "H2 is the formula for hydrogen gas.", Reasoning: "Incorrect and not relevant."
- ID: 3, Content: "CO2 is the formula for carbon dioxide.", Reasoning: "Not relevant."
- ID: 4, Content: "HO is not a correct formula.", Reasoning: "Irrelevant and incorrect."
- ID: 5, Content: "Water is a liquid essential for life.", Reasoning: "Relevant but does not directly answer the question."

**Output:**
1

**Input:**
Question: Who wrote the play "Romeo and Juliet"?
Candidates:
- ID: 1, Content: "J.K. Rowling wrote the 'Harry Potter' series.", Reasoning: "Not relevant to the question.
- ID: 2, Content: "Hemingway wrote novels like 'The Old Man and the Sea'.", Reasoning: "Not relevant to the question."
- ID: 3, Content: "Tolstoy wrote 'War and Peace'.", Reasoning: "Not relevant to the question."
- ID: 4, Content: "Jane Austen wrote 'Pride and Prejudice'.", Reasoning: "Not relevant to the question."
- ID: 5, Content: "Shakespeare is the author of 'Romeo and Juliet'.", Reasoning: "Correct and directly relevant."

**Output:**
5
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )


def rank_candidates_with_agent(agent, question: str, candidates: List[Dict], max_retries: int = 3) -> int:
    input_data = {
        "question": question,
        "candidates": candidates
    }

    input_message = json.dumps(input_data)
    messages = [{"role": "user", "content": input_message}]

    for attempt in range(max_retries):
        response = agent.generate_reply(messages)

        print(f"Attempt {attempt + 1}/{max_retries}, Raw response from LLM: {response}")

        #time.sleep(0.01)  # 添加 10 毫秒的延迟

        try:
            int_match = re.search(r"\d+", response)
            if int_match:
                top_candidate_id = int(int_match.group(0))
                if any(candidate["id"] == top_candidate_id for candidate in candidates):
                    return top_candidate_id
                else:
                    print(f"Invalid ID {top_candidate_id} not in batch. Retrying ({attempt + 1}/{max_retries})...")
            else:
                print(f"No valid integer ID found in response. Retrying ({attempt + 1}/{max_retries})...")
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Retrying ({attempt + 1}/{max_retries})...")

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
        print("--------------------------------")
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
        print("================================")
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
        {"id": 9, "relevant_content": "Paris has been the capital since the Middle Ages.", "reasoning": "Accurate and directly relevant."},
        {"id": 10, "relevant_content": "Paris is the political hub of France.", "reasoning": "Relevant and concise."},
        {"id": 11, "relevant_content": "Paris is the capital of France.", "reasoning": "Accurate and concise."},
        {"id": 12, "relevant_content": "Paris has been the capital since the Middle Ages.", "reasoning": "Accurate and directly relevant."},
        {"id": 13, "relevant_content": "Paris is the political hub of France.", "reasoning": "Relevant and concise."},
    ]

    best_candidate_id = rank(question, candidates)

    print(f"The top-ranked candidate ID: {best_candidate_id}")
