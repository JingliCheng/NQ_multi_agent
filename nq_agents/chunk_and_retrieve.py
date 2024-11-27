import tiktoken
import json
from autogen import ConversableAgent


def get_tokenizer(model_name="gpt-4"):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception as e:
        print(f"Error loading tokenizer for model {model_name}: {e}")
        return tiktoken.get_encoding("gpt-3.5-turbo")
    
def split_document(document, model_name="gpt-4", max_tokens=8192):
        tokenizer = get_tokenizer(model_name)
        tokens = tokenizer.encode(document)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end
        return chunks

def extract_content(agent, chunk, question):
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
    

def create_extract_content_agent(llm_config):
    return ConversableAgent(
        name="ExtractContentAgent",
        system_message="""
You are an agent who extracts content from a given content to answer the given question.

# Output Format
{
"reasoning": "Why this passage is relevant",
"begin_index": the begin index of the extracted passage,
"end_index": the end index of the extracted passage,
}

        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

def retrieve():
    return []
