from typing import Dict, Tuple, List
import re

import tiktoken
from rapidfuzz import fuzz, process

# from transformers import LlamaTokenizer


def text2indexed_fixed_distance(text: str, distance: int) -> str:
    words = text.split(" ") # The same logic as get_nq_tokens from google's code

    output = []
    count = 0
    for word in words:
        if count % distance == 0:
            output.append(f"<wd_idx<{count}>>")
        output.append(word)
        count += 1
    return ' '.join(output)


def indexed2text(indexed: str) -> str:
    # found the pattern "[wd_idx[some_number]] " and remove it(With space at the end)
    pattern_no_space = re.compile(r"\<wd_idx\<(\d+)\>\> ")
    pattern = re.compile(r" \<wd_idx\<(\d+)\>\> ")
    output = pattern_no_space.sub('', indexed)
    output = pattern.sub('', output)
    return output

def get_tokenizer(model_name="gpt2"):
    if 'gpt' in model_name:
        return tiktoken.encoding_for_model(model_name)
    elif 'llama' in model_name:
        return tiktoken.encoding_for_model("gpt2")
    else:
        raise ValueError(f"Tokenizer for model {model_name} not found")


def split_document(document, model_name="gpt-4", max_tokens=1000, overlap=100):
    """
    Splits a document into chunks based on [wd_idx[some_number]] markers, with a relative fixed overlap
    between the chunks. Each chunk starts with [wd_idx[some_number]] and doesn't end with one unless it
    is the last chunk.

    Parameters:
    - document (str): The document to split.
    - model_name (str): The tokenizer model name.
    - max_tokens (int): Maximum token limit for each chunk, including the overlap.
    - overlap (int): Number of tokens to overlap between chunks.

    Returns:
    - List[str]: List of chunks.
    """
    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.encode(document)
    decoded_document = tokenizer.decode(tokens)  # Ensure token alignment
    if decoded_document != document:
        raise ValueError("Token alignment failed")
    marker_pattern = r"\<wd_idx\<\d+\>\>"
    
    # Tokenize the document and locate marker positions in token space
    markers = [(match.start(), match.group()) for match in re.finditer(marker_pattern, decoded_document)]
    marker_token_positions = [len(tokenizer.encode(decoded_document[:max(m[0], 0)])) for m in markers]

    chunks = []
    i = 0

    while i < len(marker_token_positions):
        # Token start of the current chunk
        token_start = marker_token_positions[i]

        # Compute token end, with proxy_max_tokens
        token_end = token_start + max_tokens - overlap

        # Find the next marker token index after the current chunk
        next_marker_index = next(
            (j for j in range(i + 1, len(marker_token_positions)) if marker_token_positions[j] >= token_end),
            len(marker_token_positions)
        )

        # Adjust token end to avoid splitting mid-marker
        if next_marker_index < len(marker_token_positions):
            token_end = marker_token_positions[next_marker_index]
        
        # Add overlap if not the last chunk
        if next_marker_index < len(marker_token_positions) and overlap > 0:
            token_end = min(token_end + overlap, len(tokens))

        # Decode the chunk from token space back to text
        chunk_tokens = tokens[token_start:token_end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunk_text = re.sub(r"^wd_idx\<", r"<wd_idx<", chunk_text)
        chunk_text = re.sub(r"\<$", r"", chunk_text)
                
        chunks.append(chunk_text)

        # Move to the next marker
        i = next_marker_index

    return chunks

def convert_to_indexed_format(context, distance=10, model_name="llama", max_tokens=1000, overlap=100) -> Dict:
    """
    Convert the document text to indexed format and split into chunks.
    Output:
    {
        ...(other fields in the example )
        "indexed_chunks": ["chunk1", "chunk2", ...]
    }
    """
    example = context["example"]
    indexed_text = text2indexed_fixed_distance(example["document_text"], distance=distance)
    chucks = split_document(indexed_text, model_name=model_name, max_tokens=max_tokens, overlap=overlap)
    return chucks

def extract_text_from_indexes(document: str, begin_index, end_index, offset=50) -> str:
    document_list = document.split(" ")
    # Offset is the number of words to include before and after the chunk
    output = " ".join(document_list[max(0, begin_index-offset):min(len(document_list), end_index+offset)])
    return output, begin_index-offset

def grounding(context) -> List[Dict]:
    retrieved_candidates = context['retrieved_candidates']
    example = context['example']
    document = example["document_text"]

    output = []
    candidate_index = 0
    for candidate in retrieved_candidates:
        grounded_text, begin_index = extract_text_from_indexes(document, candidate["begin_index"], candidate["end_index"])
        reasoning = candidate["reasoning"]

        output.append({
            "relevant_content": grounded_text,
            "reasoning": reasoning,
            "id": candidate_index,
            "begin_index": begin_index,
        })
        candidate_index += 1

    return output


def find_long(context):
    top1_index = context['ranked_candidates']
    long_content = context['grounded_candidates'][top1_index]['relevant_content']
    return long_content


def fuzz_process(query: str, content_list: List[str], match_length: int):
    substrings = [" ".join(content_list[i:i+match_length]) for i in range(len(content_list)-match_length+1)]
    top_matches = process.extract(query, substrings, scorer=fuzz.ratio)
    best_score = top_matches[0][1]
    best_matches = [match for match in top_matches if match[1] == best_score]
    return best_matches

def fuzzy_search(query: str, content: str):
    query_length = len(query.split(" "))
    content_list = content.split(" ")

    pool = []
    for match_length in range(max(1, query_length-2), min(query_length+4, len(content_list))):
        best_matches = fuzz_process(query, content_list, match_length)
        pool.extend(best_matches)
    
    # Sort the pool by score
    pool.sort(key=lambda x: x[1], reverse=True)
    top_1 = pool[0]
    prefix = content[:top_1[2]]
    begin_index = len(prefix.split(" "))
    end_index = begin_index + len(top_1[0].split(" "))

    return begin_index, end_index


def answer2index(context, verbose=False):
    inner_begin_index, inner_end_index = fuzzy_search(context['short_answer'], context['top1_long'])
    chuck_begin = context['grounded_candidates'][context['ranked_candidates']]['begin_index']
    final_begin_index = chuck_begin + inner_begin_index
    final_end_index = chuck_begin + inner_end_index

    if verbose:
        text = context['example']['document_text']
        cut_text = ' '.join(text.split(" ")[final_begin_index:final_end_index])
        print(f"Short answer: {context['short_answer']}")
        print(f"index cut: {cut_text}")
        print(f"Final begin index: {final_begin_index}, Final end index: {final_end_index}")
        
    return final_begin_index, final_end_index

