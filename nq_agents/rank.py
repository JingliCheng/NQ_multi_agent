import requests
import json
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
        You are an intelligent assistant tasked with good reasoning ability. Now I have a task for you it is related the answer of natural question. I would like you to do this: First, please read the "Question" and
        then rank the candidate based on the relevent answer based on the relevant_content which is the reason of the answer. For your information, the input is a list of candidate answer and
        in each candidate, it contains the reasoning which is relevant_content and ID of the content. I am not asking for code, but the natural answer.

        **Question**:
        {question}

        **Candidates**:
        {candidates}

        **Output Format**:
        The ID of the top-ranked candidate as a single number, for example: 1;
        """

        # Format the candidates for the prompt
        candidates_formatted = "\n".join(
            [f"- ID: {c['id']}, Reasoning: {c['relevant_content']}" for c in candidates]
        )

        # Fill the prompt template
        prompt = prompt_template.format(question=question, candidates=candidates_formatted)

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


def rank(agent: RankAgent, extracted_contents: List[Dict], question: str) -> List[Dict]:
    """
    Rank the candidates and include the top-ranked ID for debugging.
    
    :param agent: RankAgent instance
    :param extracted_contents: List of candidates with 'relevant_content' and 'id'
    :param question: The question to rank answers for
    :return: List of ranked candidates with debug output for top ID
    """
    # Get the top-ranked candidate ID for debugging
    top_candidate_id = agent.rank_candidates(question, extracted_contents)
    
    # Return the full candidates list with the top ID for debugging purposes
    print(f"Top Candidate ID (Debug): {top_candidate_id}")  # Debug output
    print("================================================")
    return extracted_contents  # Return the original candidates for refine step


# Main function for standalone testing
if __name__ == "__main__":
    question = "What is the capital of France?"
    candidates = [
        {"relevant_content": "Paris is the capital of France.", "id": 0},
        {"relevant_content": "Marseille is a big city in France", "id": 1},
        {"relevant_content": "Paris has many different district", "id": 2}
    ]

    # Initialize the RankAgent
    rank_agent = RankAgent()

    # Test the rank function
    ranked_candidates = rank(rank_agent, candidates, question)
    print("Ranked Candidates:", ranked_candidates)