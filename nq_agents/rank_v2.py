import re
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from autogen import ConversableAgent
import time
import copy

LLM_PROVIDER = "ollama"

def get_llm_config(llm_provider=LLM_PROVIDER):
    """
    Returns the LLM configuration based on the selected provider.
    """
    llm_configs = {
        "ollama": {
            "config_list": [
                {
                    "model": "llama3.2:latest", # "llama3.2:3b-instruct-fp16",
                    "api_key": "ollama",  # Replace with actual key if necessary
                    "base_url": "http://localhost:11434/v1",
                    "temperature": 0.0,  # Use 0.0 for deterministic results
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
You are an assistant tasked with selecting the best candidate from a provided list based on a given question. Each candidate has content and an ID. 

# Task
1. Read the provided candidates carefully.
2. Rank candidate based on how likely their content is to answer the question. There could be no good candidate. But you must choose one.
3. Your output must be a JSON object with the format including the reasoning and the ID of your choice:
   {
       "reasoning": <string>,
       "top_id": <integer>
   }

# Examples
**Input:**
Question: What is the capital of France?
Candidates:
- ID: 1, Content: "Paris is the capital of France."
- ID: 2, Content: "Berlin is the capital of Germany."
- ID: 3, Content: "Lyon is a city in France."

**Output:**
{
    "reasoning": "ID 1 is the best candidate because it is a statement directly answering the question.",
    "top_id": 1
}

**Input:**
Question: What is the largest planet in our solar system?
Candidates:
- Content: "Mars is the largest planet in our solar system.", ID: 23
- Content: "Earth is the third planet from the Sun.", ID: 24
- Content: "Jupiter is the largest planet in our solar system.", ID: 25


**Output:**
{
    "reasoning": "ID 25 is the best candidate because it is a statement directly answering the question.",
    "top_id": 25
}
""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

def rank_candidates_with_agent(agent, question: str, candidates: List[Dict], max_retries: int = 1) -> int:
    user_message = f"Question: {question}\n"
    for candidate in candidates:
        user_message += f"- Content: {candidate['relevant_content']}, ID: {candidate['id']}\n"

    messages = [{"role": "user", "content": user_message}]  # Ensure the format matches LLM expectations.

    for attempt in range(max_retries):
        print("1")
        response = agent.generate_reply(messages)
        print("2")

        if not response:

            print(f"Attempt {attempt + 1}/{max_retries}: LLM response is empty. Retrying...")
            continue
        print("================================")
        print(f"Attempt {attempt + 1}/{max_retries} ")
        print(f"Raw response from LLM: {response}")
        candidate_ids = [candidate["id"] for candidate in candidates]
        print(f"Processing candidates with IDs: {candidate_ids}")
        try:
            pattern = r'"top_id": (\d+)'
            match = re.search(pattern, response)
            if match:
                top_candidate_id = int(match.group(1))

            if top_candidate_id and top_candidate_id in candidate_ids:
                return top_candidate_id
            else:
                print(f"Invalid ID {top_candidate_id} not in batch. Retrying ({attempt + 1}/{max_retries})...")
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Retrying ({attempt + 1}/{max_retries})...")

    print(f"Failed to get a valid ID after {max_retries} attempts.")
    return None

def process_batch(rank_agent, question, batch):
    print("Processing batch: ", batch)
    print('batch size: ', len(batch))
    top_candidate_id = rank_candidates_with_agent(rank_agent, question, batch)
    print(f"Top candidate ID: {top_candidate_id}")
    for candidate in batch:
        if candidate["id"] == top_candidate_id:
            return candidate
    batch_ids = list(map(lambda x: x["id"], batch))
    print(f"No valid candidate found in batch{batch_ids}")
    return None

def rank(context: Dict, batch_size: int = 3) -> int:
    candidates =  copy.deepcopy(context["grounded_candidates"])
    question = context['example']['question_text']
    round_number = 1

    while len(candidates) > 1:
        print(f"Round {round_number}: {len(candidates)} candidates remaining.")
        next_round_candidates = []

        estimated_agents_number = len(candidates)//batch_size + 1
        rank_agents = [create_rank_agent() for _ in range(estimated_agents_number)]

        print('# of rank_agents: ', len(rank_agents))


        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_batch, rank_agents[i//batch_size], question, candidates[i:i + batch_size])
                for i in range(0, len(candidates), batch_size)
            ]
            for future in futures:
                result = future.result()
                if result:
                    next_round_candidates.append(result)

        if next_round_candidates:
            candidates = list(filter(lambda x: x is not None, next_round_candidates))
        else:
            print("Error: All batches were skipped due to invalid LLM outputs. Roll back to previous candidates.")
            # roll back to previous candidates, so that the first candidate is in the previous round is returned.
            break
        round_number += 1

    print(f"# of candidates in rank pool: {len(candidates)}")
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
