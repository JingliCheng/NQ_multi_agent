import re
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from autogen import ConversableAgent
import time

LLM_PROVIDER = "ollama"

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
    }

    if llm_provider not in llm_configs:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    return llm_configs[llm_provider]

def create_rank_agent():
    llm_config = get_llm_config()
    return ConversableAgent(
        name="RankAgent",
        system_message="""
You are an assistant tasked with selecting the best candidate from a provided list based on a given question. Each candidate has content, reasoning, and metadata including an ID and begin_index. 

# Task
1. Read the provided candidates carefully, including their metadata.
2. Evaluate the relevance of each candidate's reasoning and content to the question.
3. Choose the best candidate by selecting the ID from the provided list. Do not use your own knowledge—only rely on the given candidates and their reasoning.
4. Your output must be a JSON object with the format:
   {
       "top_id": <integer>
   }
5. Do not create new IDs, provide explanations, or include extra text—only return the JSON object with the selected ID.
6. The ID could be from 1-100.
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
    messages = [{"role": "user", "content": input_message}]  # Ensure the format matches LLM expectations.

    for attempt in range(max_retries):
        response = agent.generate_reply(messages)

        if not response:

            print(f"Attempt {attempt + 1}/{max_retries}: LLM response is empty. Retrying...")
            continue
        print("================================")
        print(f"Attempt {attempt + 1}/{max_retries} ")
        #print(f"Raw response from LLM: {response}")
        candidate_ids = [candidate["id"] for candidate in candidates]
        print(f"Processing candidates with IDs: {candidate_ids}")
        try:
            # Parse the response as JSON to extract the "top_id" field
            response_json = json.loads(response)
            top_candidate_id = response_json.get("top_id")

            if top_candidate_id and any(candidate["id"] == top_candidate_id for candidate in candidates):
                return top_candidate_id
            else:
                print(f"Invalid ID {top_candidate_id} not in batch. Retrying ({attempt + 1}/{max_retries})...")
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Retrying ({attempt + 1}/{max_retries})...")

    print(f"Failed to get a valid ID after {max_retries} attempts.")
    return None

def process_batch(rank_agent, question, batch):
    try:
        top_candidate_id = rank_candidates_with_agent(rank_agent, question, batch)
        print(f"Top candidate ID: {top_candidate_id}")
        return next(candidate for candidate in batch if candidate["id"] == top_candidate_id)
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
                if result:
                    next_round_candidates.append(result)

        if not next_round_candidates:
            print("Error: All batches were skipped due to invalid LLM outputs.")
            return None

        candidates = next_round_candidates
        round_number += 1

    return candidates[0]["id"] if candidates else None

if __name__ == "__main__":
    question = "What is the capital of France?"
    candidates = [
    { "relevant_content": "Nice is a city in southern France.", "reasoning": "Not the capital.", "id": 1, "begin_index": 20 },
    { "relevant_content": "Berlin is the capital of Germany.", "reasoning": "Irrelevant to the question.", "id": 2, "begin_index": 4 },
    { "relevant_content": "Lyon is a city in France.", "reasoning": "Not the capital but relevant.", "id": 3, "begin_index": 54 },
    { "relevant_content": "Madrid is the capital of Spain.", "reasoning": "Not relevant to France.", "id": 4, "begin_index": 67 },
    { "relevant_content": "Marseille is a major city in France.", "reasoning": "Not the capital.", "id": 5, "begin_index": 45 },
    { "relevant_content": "Paris is home to the Eiffel Tower.", "reasoning": "Relevant but not directly answering.", "id": 6, "begin_index": 87 },
    { "relevant_content": "Rome is the capital of Italy.", "reasoning": "Irrelevant to the question.", "id": 7, "begin_index": 232 },
    { "relevant_content": "Nice is a city in southern France.", "reasoning": "Not the capital.", "id": 8, "begin_index": 25 },
    { "relevant_content": "Paris has been the capital since the Middle Ages.", "reasoning": "Accurate and directly relevant.", "id": 9, "begin_index": 2 },
    { "relevant_content": "Paris is the political hub of France.", "reasoning": "Relevant and concise.", "id": 10, "begin_index": 25 },
    { "relevant_content": "Paris is the capital of France.", "reasoning": "Accurate and concise.", "id": 11, "begin_index": 3 },
    { "relevant_content": "Paris has been the capital since the Middle Ages.", "reasoning": "Accurate and directly relevant.", "id": 12, "begin_index": 47 },
    { "relevant_content": "Paris is the political hub of France.", "reasoning": "Relevant and concise.", "id": 13, "begin_index": 896 }
]

    best_candidate_id = rank(question, candidates)
    print(f"The top-ranked candidate ID: {best_candidate_id}")
