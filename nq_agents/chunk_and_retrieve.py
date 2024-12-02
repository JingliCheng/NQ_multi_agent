import ratelimit
import tiktoken
from autogen import ConversableAgent
import time
import json
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry
from ratelimit.exception import RateLimitException

# QPS and concurrency limits
MAX_QPS = 2  
MAX_CONCURRENT_REQUESTS = 2  

# Token limit for different models
DEFAULT_TOKEN_LIMIT = 8192  # default token limit）
LLAMA_TOKEN_LIMIT = 20480   # llama model token limit
OPENAI_TOKEN_LIMIT = 20480 # OpenAI GPT-4 Turbo model token limit

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

    # Validate the selected provider
    if llm_provider not in llm_configs:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    # Return the configuration for the selected provider
    return llm_configs[llm_provider]

def create_extract_content_agent():
    # print("Creating ExtractContentAgent")
    llm_config = get_llm_config()
    return ConversableAgent(
        name="ExtractContentAgent",
        system_message="""
# Role

You are an intelligent assistant designed to analyze and interpret structured text documents. Your main task is to carefully review and evaluate the provided content to identify specific sections that directly answer the question or contain the relevant information. You should thoughtfully analyze the text and ensure your results are accurate and relevant to the task.
 
# Instruction

The document is structured as follows: Each section is marked by `<wd_idx<i>>` and `<wd_idx<j>>`, where `i` and `j` represent the starting word indices of the text within the markers.
 
You **MUST**:

0. DONT ANSWER THE QUESTION

1. **Word Index Usage**: While analyzing and extracting the content, do not use word index markers to filter or prioritize the context.

2. **Reasoning**: Provide a short and concise reasoning process that explains why the extracted content is relevant to the question.

3. **Formatting**: Your output must strictly follow the specified format.

4. Categorize your response based on the following scenarios:

   - **Case 1:** If you find a single section that meets the requirements, output the corresponding `i` and `j` values.

   - **Case 2:** If you find multiple sections that could meet the requirements, output the `i` and `j` values for each section clearly.

   - **Case 3:** If no section meets the requirements, explicitly indicate that no suitable section was found by outputting `i = none` and `j = none`.
 
# Input Format
[Question] Question Text  
[Context] Context Text with structured markers like <wd_idx<0>>

# Output Format
{
    "found": true/false,
    "candidates": [
        {
            "reasoning": "Reasoning Text",
            "begin_index": start_index,
            "end_index": end_index
        },
        {
            "reasoning": "Reasoning Text 2",
            "begin_index": start_index_2,
            "end_index": end_index_2
        }
    ]
}


# Examples

## Example 1: (Case 1: Single Match)

**Input:**  
[Question] Why is email marketing considered cost-effective?
[Context] <wd_idx<10>> Email marketing is often reported as second only to search <wd_idx<20>> marketing as the most effective online marketing tactic. Email marketing <wd_idx<30>> is significantly cheaper and faster than traditional mail, mainly because <wd_idx<40>> of the high cost and time required in a traditional <wd_idx<50>> mail campaign for producing the artwork, printing, addressing, and mailing. <wd_idx<60>> Businesses and organizations who send a high volume of emails <wd_idx<70>> can use an ESP (email service provider) to gather information <wd_idx<80>> about the behavior of the recipients. The insights provided by <wd_idx<90>> consumer response to email marketing help businesses and organizations understand <wd_idx<100>> and make use of consumer behavior. Email provides a cost-effective <wd_idx<110>> method to test different marketing content, including visual, creative, marketing <wd_idx<120>>

**Output:**  
{
    "found": true,
    "candidates": [
        {
            "reasoning": "This passage explains that email marketing provides insights about consumer behavior by analyzing recipient responses.",
            "begin_index": 80,
            "end_index": 100
        },
    ]
}

        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

def retrieve(example, verbose=False):
    """
    Perform extraction for all indexed chunks in the example using an agent.
    Consolidates the results into the specified JSON format with `found` and `candidates`.
    """
    # Extract indexed chunks from the example
    question = example.get("question_text", "")
    document = example.get("document_text", "")

    print(question)
    indexed_chunks = example.get("indexed_chunks", [])
    if not indexed_chunks:
        return {"found": False, "candidates": []}
    # TODO: remove
    print(f"Extracting content from {len(indexed_chunks)} indexed chunks\n")
    print(f"Example.indexed_chunks: {indexed_chunks}")

    # Step 1: Create ExtractContentAgent for each chunk
    extract_agents = [create_extract_content_agent() for _ in indexed_chunks]

    # Step 2: Define rate-limited content extraction function
    @sleep_and_retry
    @limits(calls=MAX_QPS, period=1)
    def extract_content_with_rate_limit(agent, chunk, question):
        while True:
            try:
                return extract_content(agent, chunk, question)
            except RateLimitException:
                time.sleep(1)

    # Step 3: Extract content from each chunk
    extracted_contents = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        print("Submitting extraction tasks")
        print("question", question)
        print("document", indexed_chunks)
        # print("\n\n\n")
        future_to_chunk = {
            executor.submit(
                extract_content_with_rate_limit, agent, chunk, question
            ): chunk
            for agent, chunk in zip(extract_agents, indexed_chunks)
        }
        for future in future_to_chunk:
            try:
                result = future.result()
                extracted_contents.append(result)
                if verbose:
                    print(f"Extracted result: {result}")
            except Exception as e:
                if verbose:
                    print(f"Error during extraction: {e}")
                extracted_contents.append({"found": False, "candidates": []})

# Consolidate results into final JSON format
    extract_candidates = []

    for content in extracted_contents:

        if content.get("found", False):
            for candidate in content.get("candidates", []):
                try:
                    # Validate and extract text based on begin_index and end_index
                    begin_index = int(candidate.get("begin_index", 0))
                    end_index = int(candidate.get("end_index", 0))
                    
                    # Check if indices are valid and within document bounds
                    if begin_index >= 0 and end_index > begin_index and end_index <= len(document.split()):
                        # Extract grounded text from document
                        grounded_text = extract_text_from_indexes(document, begin_index, end_index)
                        candidate["grounded_text"] = grounded_text
                        extract_candidates.append(candidate)
                    else:
                        if verbose:
                            print(f"Invalid indices for candidate: {candidate}")
                except Exception as e:
                    if verbose:
                        print(f"Error processing candidate: {e}")
    
    print("final candidates", extract_candidates)

    final_output = {
        "found": len(extract_candidates) > 0,
        "candidates": extract_candidates,
    }

    # Print consolidated output if verbose
    if verbose:
        print(f"Final consolidated output: {json.dumps(final_output, indent=2)}")

    return final_output

def extract_content(agent, chunk, question):
    input_message = f"[Question] {question}\n[Context] {chunk}"
    response = agent.generate_reply(
        messages=[{"content": input_message, "role": "user"}]
    )
    print("\n\n\nRaw Response:", response)
    print("------------------")

    try:

        # response_content = response["messages"][-1]["content"]
        if isinstance(response, dict):  # If the response is already a dictionary
            response_json = response
        elif isinstance(response, str):  # If the response is a string, parse it
            response_json = json.loads(response)
        else:
            raise ValueError("Unexpected response format. Expected dict or JSON string.")

        print("*"*100)
        print("response_json", response_json)
        
        # Check if relevant content was found
        if response_json.get("found", False):
            candidates = response_json.get("candidates", [])
            return {"found": True, "candidates": candidates}
        else:
            # No relevant content found
            return {"found": False, "candidates": []}
    except Exception as e:
        # Handle parsing or other exceptions gracefully
        return {"found": False, "candidates": [], "error": str(e)}
    
def extract_text_from_indexes(document_text, begin_index, end_index):
    """
    Extract text from the document based on word indexes.

    Args:
        document_text (str): The full document text.
        begin_index (int): The starting word index.
        end_index (int): The ending word index.

    Returns:
        str: The extracted text from the document.
    """
    try:
        # Split the document into words using spaces
        words = document_text.split()
        
        # Extract the specified range of words
        extracted_text = " ".join(words[begin_index:end_index])

        return extracted_text
    except Exception as e:
        print(f"Error while extracting text from indexes: {e}")
        return ""