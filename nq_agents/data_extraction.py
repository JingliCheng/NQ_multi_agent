import json
import random
import os
from tqdm import tqdm


TRAIN_FILE = '.local_data/simplified-nq-train.jsonl'
DEV_FILE = '.local_data/v1.0-simplified_nq-dev-all.jsonl'


def count_lines(file_path):
    """Count the number of effective lines in a large file, accounting for an optional final newline."""
    def blocks(file, size=65536):
        while True:
            b = file.read(size)
            if not b:
                break
            yield b

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        # Sum all newlines
        line_count = sum(bl.count("\n") for bl in blocks(f))
        
        # Check if the file ends without a newline
        f.seek(0, 2)  # Go to the end of the file
        if f.tell() > 0:  # File is non-empty
            f.seek(f.tell() - 1)  # Move back one character from the end
            last_char = f.read(1)
            if last_char != "\n":
                line_count += 1  # Count the last line if it doesn't end with a newline

    return line_count

def read_first_lines(file_path, num_lines=5):
    """Read the first num_lines lines from a file."""
    output = []
    with open(file_path, 'r') as file:
        for _ in range(num_lines):
            output.append(json.loads(file.readline().strip()))
    return output

def list_to_jsonl(data_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')

def write_jsonl(examples, output_path):
    """Write examples to a jsonl file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

def sample_examples(file_path, output_path=None, k=10, seed=42):
    """Sample k examples from input jsonl and write to output jsonl."""
    random.seed(seed)
    output = []
    # Open the file and iterate with an unbounded progress bar
    with open(file_path, 'r', encoding='utf-8') as file:
        # Initialize tqdm without a total, to track processed lines
        for i, line in enumerate(tqdm(file, desc="Sampling Lines", unit="line")):
            if i < k:
                output.append(json.loads(line.strip()))
            else:
                # Replace elements in the reservoir with decreasing probability
                j = random.randint(0, i)
                if j < k:
                    output[j] = json.loads(line.strip())
    # Set the output path if not provided
    if output_path is None:
        output_path = file_path.replace('.jsonl', f'_sample{k}_seed{seed}.jsonl')
    # Write sampled output to the output JSONL file
    write_jsonl(output, output_path)
    return output

def keep_short_ans_examples(file_path, output_path=None):
    """Keep examples with short answer."""
    output = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # Initialize tqdm without a total count
        for line in tqdm(file, desc="Filtering Short Answer Examples", unit="line"):
            data = json.loads(line.strip())
            if (data['annotations'][0]['short_answers'] 
                or data['annotations'][0]['yes_no_answer'] != 'NONE'):
                output.append(data)
                
    # Set the output path if not provided
    if output_path is None:
        output_path = file_path.replace('.jsonl', '_short_ans.jsonl')
    
    # Write the filtered output to the output JSONL file
    write_jsonl(output, output_path)
    return output

def sample_short_ans_examples(file_path, output_path=None, k=100, seed=42):
    """Sample k examples with short answer from input jsonl and write to output jsonl."""
    temp_path = file_path.replace('.jsonl', '_temp.jsonl')
    keep_short_ans_examples(file_path, temp_path)
    if output_path is None:
        output_path = file_path.replace('.jsonl', f'_sample{k}_seed{seed}.jsonl')
    output = sample_examples(temp_path, output_path, k, seed)
    os.remove(temp_path)
    return output
