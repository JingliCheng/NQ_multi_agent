import requests
import json


class OllamaAgent:
    def __init__(self, api_url="http://127.0.0.1:11434/api/generate", model="llama3.2:latest"):
        """
        Initialize Ollama Agent.
        
        :param api_url: Ollama API address
        :param model: model name
        """
        self.api_url = api_url
        self.model = model

    def send_request(self, question, long_answer, wrong_answer, temperature=0.7, max_tokens=10000):
        """
        send request to Ollama API to generate short answer
        """
        # Define prompt 
        prompt_template = """
        You are an intelligent assistant. Your task is to extract a concise and accurate answer to a given question from the provided long answer. Follow these instructions carefully:

        ## Key Guidelines:
        1. **Focus on Extraction**: YOU MUST ONLY RESPONSE WITH SHORT ANSWER. The answer must be directly and exactly taken from the **Long Answer** without modification or rephrasing.
        2. **Answer Format**: 
        - The answer should be as short as possible, ideally just a few words or a single word.
        - Avoid sentences unless explicitly required.
        3. **Avoid Wrong Answer**: Do not include any parts of the text related to {wrong_answer}.
        4. **No Answer Found**: If the **Long Answer** does not contain relevant information, output "None".
        5. **Strictly Use Provided Text**: Do not invent, summarize, or interpret new information outside of what is given.
        6. **Formatting**: Your output should strictly follow the specified format.

        ---

        ## Step-by-Step Thought Process:
        1. **Understand the Question**: Carefully read and comprehend the given question.
        2. **Locate Relevant Text**: Identify the part of the **Long Answer** that directly addresses the question.
        3. **Verify Accuracy**: Ensure the selected text answers the question precisely without including irrelevant details.
        4. **Exclude Incorrect Parts**: Check that the selected text avoids the context related to {wrong_answer}.
        5. **Extract Short Answer**: Extract the minimal text required to answer the question directly.
        6. **Handle No Answer Cases**: If no suitable text is found, output "None".

        ---

        ## Examples

        ### Example 1:
        [Input]
        **Question**:  
        When are hops added to the brewing process?  

        **Long Answer**:  
        Hops are added during the boiling stage, where they release flavors, bitterness, and aroma, while sterilizing the wort and evaporating off-flavors. This process typically lasts 45 to 90 minutes.
        
        [Output]
        The boiling process  

        ### Example 2:
        [Input]
        **Question**:
        Where is the capital of France?

        **Long Answer**: 
        Paris is the capital of France. It is renowned for its history, art, and culture, often referred to as the “City of Light.” Paris is not only a political center but also a global symbol of elegance and sophistication.
        
        [Output]
        Paris

        ### Example 3:
        [Input]
         **Question**:
        What is the use of jdk in java
        
        **Long Answer**: 
        applications </Li> <Li> jarsigner -- the jar signing and verification tool </Li> <Li> javah -- the C header and stub generator , used to write native methods </Li> <Li> javap -- the class file disassembler </Li> <Li> javaws -- the Java Web Start launcher for JNLP applications </Li> <Li> JConsole -- Java Monitoring and Management Console </Li> <Li> jdb -- the debugger </Li> <Li> jhat -- Java Heap Analysis Tool ( experimental ) </Li> <Li> jinfo -- This utility gets configuration information from a running Java process or crash dump . ( experimental ) </Li> <Li> jmap Oracle jmap - Memory Map -- This utility outputs the memory map for
        
        [Output]
        Development and debugging
        
        ---
        # NOW YOU TRY TO GENERATE SHORT ANSWER FOR THE GIVEN QUESTION AND LONG ANSWER 
        [Input]
        **Question**:  
        {question}

        **Long Answer**:  
        {long_answer}
        """

        # fill template
        prompt = prompt_template.format(
            question=question, 
            long_answer=long_answer, 
            wrong_answer=wrong_answer if wrong_answer else ""
        )
        print("prompt:", prompt)
        # construct API request
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            # parse the response and return the short answer
            lines = response.text.splitlines()
            result = []
            for line in lines:
                try:
                    json_line = json.loads(line)
                    if "response" in json_line:
                        result.append(json_line["response"])
                except json.JSONDecodeError:
                    pass
            
            return "".join(result).strip()  # concat results

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama API: {e}")

def refine(question: str, long_answer: str, wrong_answer: str = ""):
    agent = OllamaAgent()
    try:
        short_answer = agent.send_request(question, long_answer, wrong_answer)
        return short_answer
    except Exception as e:
        return e

# Simple test with main
if __name__ == "__main__":
    question = "what is the widest highway in north america"
    long_answer = """
    King's Highway 401, commonly referred to as Highway 401 and also known by its official name as the Macdonald–Cartier Freeway or colloquially as the four-oh-one,[3] is a controlled-access400-series highway in the Canadian province of Ontario. It stretches 828.0 kilometres (514.5 mi) from Windsor in the west to the Ontario–Quebec border in the east. The part of Highway 401 that passes through Toronto is North America's busiest highway,[4][5] and one of the widest.[6][7] Together with Quebec Autoroute 20, it forms the road transportation backbone of the Quebec City–Windsor Corridor, along which over half of Canada's population resides and is also a Core Route in the National Highway System of Canada. The route is maintained by the Ministry of Transportation of Ontario (MTO) and patrolled by the Ontario Provincial Police. The speed limit is 100 km/h (62 mph) throughout its length, unless posted otherwise.
    """
    wrong_answer = "401"
    print("short answer:", refine(question, long_answer, wrong_answer))





# def create_refine_answer_agent(llm_config):
#     return ConversableAgent(
#         name="RefineAnswerAgent",
#         system_message="""
# You are an agent who refines the given candidate answers to answer the given question.
# Output:
# {
# "refined_answer": "refined answer"
# }
#         """,
#         llm_config=llm_config,
#         human_input_mode="NEVER",
#     )

# def refine(agent, ranked_candidates, question) -> Tuple[str, float]:
#     return ('', 0.5)

# def refine(agent, long_answer: str, question: str) -> Tuple[str, float]:
    
#     """
#     Uses the RefineAnswerAgent to generate a concise short answer from a long answer and a question.

#     Args:
#         agent: The RefineAnswerAgent created by create_refine_answer_agent.
#         long_answer (str): The detailed answer.
#         question (str): The question to be answered.

#     Returns:
#         str: The refined short answer.
#     """
#     if not long_answer or not question:
#         return ("Invalid input. Please provide both a long answer and a question.", 0.5)