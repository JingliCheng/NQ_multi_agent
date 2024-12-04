import requests
import json
import re
from typing import List, Dict


class RankAgent:
    def __init__(self, api_url="http://127.0.0.1:11434/api/generate", model="llama3.2:latest"):
        """
        Initialize the RankAgent.
        
        :param api_url: API address for the LLM
        :param model: Model name
        """
        self.api_url = api_url
        self.model = model

    def rank_candidates(self, question, candidates, temperature=0.0, max_tokens=512):
        """
        Send a request to rank candidates and return the ID of the top-ranked candidate.
        """
        # Define the prompt
        prompt_template = """
        You are an intelligent assistant with excellent reasoning skills. Your task is to analyze a question and evaluate a list of up to 100 candidate answers to select the most relevant one based on the provided reasoning and content.

        For each candidate, you will consider:
        1. The relevance of the content to the question.
        2. The quality and logic of the reasoning provided.

        **Instructions**:
        - Carefully review the "Question" and the list of "Candidates."
        - Evaluate each candidate's "Content" and its corresponding "Reasoning" to determine the best match.
        - Select only one candidate as the top-ranked answer.
        - Provide the ID of the top-ranked candidate in the following format: `ID: <number>` (e.g., `ID: 1`).
        - After the ID, briefly explain your reasoning for selecting this candidate (e.g., why it is the most relevant).

        **Question**:
        {question}

        **Candidates**:
        {candidates}

        **Output Format**:
        - First, provide the ID of the top-ranked candidate in the format: `ID: <number>` (e.g., `ID: 1`).
        - Then, provide a concise explanation (e.g., "This candidate is the most relevant because...").
        """


        # Format the candidates for the prompt, including reasoning for each
        candidates_formatted = "\n".join(
            [
                f"- ID: {c['id']}, Content: {c['relevant_content']}, Reasoning: {c['reasoning']}"
                for c in candidates
            ]
        )

        # Use a default reasoning if not explicitly required
        reasoning = "Please evaluate the relevance and reasoning quality based on the content provided for each candidate."

        # Fill the prompt template
        prompt = prompt_template.format(
            question=question,
            candidates=candidates_formatted,
            reasoning=reasoning
        )

        # Construct API request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()

            # Parse the response to extract the top candidate ID
            lines = response.text.splitlines()
            result = []
            for line in lines:
                try:
                    json_line = json.loads(line)
                    if "response" in json_line:
                        result.append(json_line["response"])
                except json.JSONDecodeError:
                    pass

            return "".join(result).strip()  # Extract the ID as a single number
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to the API: {e}")



def rank(extracted_contents: List[Dict], question: str, failed_questions: List[Dict]) -> Dict:
    """
    Rank the candidates and return only the top-ranked candidate.
    
    :param agent: RankAgent instance
    :param extracted_contents: List of candidates with 'relevant_content' and 'id'
    :param question: The question to rank answers for
    :return: Top-ranked candidate (as a dictionary)
    """
    rank_agent = RankAgent()
    retries = 5
    # # Get the raw response from the LLM
    # raw_response = rank_agent.rank_candidates(question, extracted_contents)

    # # Debug: print the raw response from the LLM
    # print(f"Raw Response from LLM:\n{raw_response}")
    # print("================================================")

    # # Updated part of the rank function
    # match = re.search(r"(\d+)", raw_response)  # Modified regex to capture any standalone number
    # if not match:
    #     raise ValueError(f"Unable to extract top candidate ID from LLM response: {raw_response}")

        # # Convert the top candidate ID to an integer
    # top_candidate_id = int(match.group(1))
    # # Find and return the top-ranked candidate
    # top_candidate = next(candidate for candidate in extracted_contents if candidate['id'] == top_candidate_id)
    # # Extract the relevant content of the top-ranked candidate
    # top_candidate_content = top_candidate['relevant_content']


    # # Debug: Print the top candidate ID
    # print(f"Top Candidate ID (Debug): {top_candidate_id}")
    # print("================================================")
    # # Debug output
    # print(f"Top Candidate (Parsed): {top_candidate}")
    # print("================================================")
    # print(f"Print answer string:.{top_candidate_content}")
    # return top_candidate_id
    for attempt in range(retries):
        try:
            raw_response = rank_agent.rank_candidates(question, extracted_contents)
            print(f"Raw Response from LLM (Attempt {attempt + 1}):\n{raw_response}")
            
            # Check if the response contains a valid ID
            match = re.search(r"ID: (\d+)", raw_response)
            if match:
                top_candidate_id = int(match.group(1))
                top_candidate = next(
                    candidate for candidate in extracted_contents if candidate['id'] == top_candidate_id
                )
                print(f"Top Candidate ID: {top_candidate_id}")
                #print(f"Top Candidate Content: {top_candidate['relevant_content']}")
                return top_candidate_id
            else:
                print("Invalid format. Reinforcing prompt...")

        except Exception as e:
            print(f"Error: {e}")

        if attempt < retries - 1:
            print(f"Retrying... (Attempt {attempt + 2}/{retries})")
    print("================================================")

    # If all retries fail, log the failed question and candidates
    print("Max retries reached. Logging failed question.")
    failed_entry = {"question": question, "candidates": extracted_contents}
    failed_questions.append(failed_entry)

    # Optionally, log to a file
    with open("failed_questions_log.json", "a") as log_file:
        json.dump(failed_entry, log_file)
        log_file.write("\n")

    return {"id": 0, "relevant_content": "No valid response", "reasoning": "Skipped due to invalid responses."}





# Main function for standalone testing
if __name__ == "__main__":
    question = "What is the capital of France?"
    candidates = [
        {"id": 1, "relevant_content": "Paris is the capital of France.", "reasoning": "Paris is the political and cultural hub of France."},
        {"id": 2, "relevant_content": "Berlin is the capital of Germany.", "reasoning": "Berlin is a major European city but not the capital of France."},
        {"id": 3, "relevant_content": "Lyon is a city in France.", "reasoning": "Lyon is significant but not the capital."}
    ]

    # Initialize the RankAgent
    # List to store failed questions
    failed_questions = []

    ranked_candidate_id = rank(candidates, question, failed_questions)

    if ranked_candidate_id != 0:
        print(f"Top-ranked candidate ID: {ranked_candidate_id}")
    else:
        print("No valid candidate could be ranked.")

    # Print logged failed questions
    if failed_questions:
        print("Failed Questions Log:")
        print(json.dumps(failed_questions, indent=4))

