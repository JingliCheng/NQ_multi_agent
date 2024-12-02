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
        You are an intelligent assistant tasked with good reasoning ability. Now I have a task for you it is related to the answer of a natural question. I would like you to do this: First, please read the "Question" and
        then rank the relevant_candidate based on reasoning which is the reason of the answer and also the relevant_candidate itself. For your information, the input is a list of candidate answers, and
        in each candidate, it contains the relevant_candidate, reasoning, and ID of the content. I am not asking for code, but the natural answer.

        **Question**:
        {question}

        **Candidates**:
        {candidates}
        
        **Reasoning**:
        {reasoning}
        
        **Output Format**:
        The ID of the top-ranked candidate as a single number, for example: 1;
        """

        # Format the candidates for the prompt
        candidates_formatted = "\n".join(
            [f"- ID: {c['id']}, Content: {c['relevant_content']}" for c in candidates]
        )
        # Use a default reasoning if not explicitly required
        reasoning = "Please evaluate reasoning quality based on the answer content."

        # Fill the prompt template
        prompt = prompt_template.format(question=question, candidates=candidates_formatted, reasoning=reasoning)

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



def rank(extracted_contents: List[Dict], question: str) -> Dict:
    """
    Rank the candidates and return only the top-ranked candidate.
    
    :param agent: RankAgent instance
    :param extracted_contents: List of candidates with 'relevant_content' and 'id'
    :param question: The question to rank answers for
    :return: Top-ranked candidate (as a dictionary)
    """
    rank_agent = RankAgent()
    # Get the raw response from the LLM
    raw_response = rank_agent.rank_candidates(question, extracted_contents)

    # Debug: print the raw response from the LLM
    print(f"Raw Response from LLM:\n{raw_response}")
    print("================================================")

    # Extract the top candidate ID from the response
    match = re.search(r"ID:\s*(\d+)", raw_response)
    if not match:
        raise ValueError("Unable to extract top candidate ID from LLM response.")

    # Convert the top candidate ID to an integer
    top_candidate_id = int(match.group(1))
    # Find and return the top-ranked candidate
    top_candidate = next(candidate for candidate in extracted_contents if candidate['id'] == top_candidate_id)
    # Extract the relevant content of the top-ranked candidate
    top_candidate_content = top_candidate['relevant_content']


    # Debug: Print the top candidate ID
    print(f"Top Candidate ID (Debug): {top_candidate_id}")
    print("================================================")
    # Debug output
    print(f"Top Candidate (Parsed): {top_candidate}")
    print("================================================")
    print(f"Print answer string:.{top_candidate_content}")
    return top_candidate_id


# Main function for standalone testing
if __name__ == "__main__":
    question = "What is the capital of France?"
    candidates = [
        {"relevant_content": "Paris is the capital of France.","reasoning":"test", "id": 5},
        {"relevant_content": "Marseille is a big city in France","reasoning":"test", "id": 1},
        {"relevant_content": "Paris has many different district","reasoning":"test", "id": 2}
    ]

    # Initialize the RankAgent


    # Test the rank function
    ranked_candidates = rank(candidates, question)
    print("Ranked Candidates:", ranked_candidates)

