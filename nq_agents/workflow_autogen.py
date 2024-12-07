import os
import openai
from typing import Dict, Tuple
import json
import time
import re
from ratelimit import limits, sleep_and_retry, RateLimitException
from concurrent.futures import ThreadPoolExecutor
from abc import ABC
# from autogen import ConversableAgent
# Import BaseAgentSystem from Original Repo
from nq_agents.multi_agent import BaseAgentSystem, get_short_answers, TimeLogger

from nq_agents import indexing
from nq_agents import chunk_and_retrieve
from nq_agents import rank_v2
from nq_agents import refine


# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "placeholder" # Replace with your actual API key

# QPS and concurrency limits
MAX_QPS = 15  # 每秒最多请求数
MAX_CONCURRENT_REQUESTS = 15  # 最大并发请求数

# Token limit for different models
DEFAULT_TOKEN_LIMIT = 8192  # 默认 Token 限制（8k）
LLAMA_TOKEN_LIMIT = 20480   # Llama 模型 Token 限制（80k）
OPENAI_TOKEN_LIMIT = 20480 # OpenAI GPT-4 Turbo 模型 Token 限制（100k）


# Load data from file
TRAIN_FILE_100 = 'data/v1.0-simplified_nq-dev-all_sample100_seed42.jsonl'
DEV_FILE_100 = 'data/v1.0-simplified_nq-dev-all_sample100_seed42.jsonl'
SINGLE_ENTRY_FILE= 'data/first_entry_sample.jsonl'

# Configure LLM provider
LLM_PROVIDER = "ollama"
OLLAMA_API_BASE = "http://localhost:11434/v1"
MAX_TOKENS = LLAMA_TOKEN_LIMIT if LLM_PROVIDER == "ollama" else OPENAI_TOKEN_LIMIT


class WorkflowAutogen(BaseAgentSystem):
    def __init__(
        self,
        llm_provider="ollama",
        api_key=None,
        max_tokens=20480,
        max_qps=MAX_QPS,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
    ):
        """
        Initialize the multi-agent autogen system.

        Args:
            llm_provider (str): The LLM provider, e.g., 'openai' or 'ollama'.
            api_key (str): API key for the LLM provider.
            max_tokens (int): Maximum tokens for the LLM.
            max_qps (int): Maximum queries per second.
            max_concurrent_requests (int): Maximum concurrent requests.
        """
        super().__init__()
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.max_qps = max_qps
        self.max_concurrent_requests = max_concurrent_requests

        # Set up LLM configuration
        self.llm_config = self.get_llm_config()

    def get_llm_config(self):
        llm_configs = {
            "ollama": {
                "config_list": [
                    {
                        "model": "llama3.2:latest",
                        "api_key": "ollama",
                        "base_url": "http://localhost:11434/v1",
                        "temperature": 0.7,
                    }
                ]
            },
            "openai": {
                "config_list": [
                    {
                        "model": "gpt-4o",
                        "api_key": self.api_key,
                        "temperature": 0.7,
                    }
                ]
            },
        }
        if self.llm_provider not in llm_configs:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        return llm_configs[self.llm_provider]
    

    def predict(self, example: Dict, verbose: bool = False) -> Tuple[str, float]:
        # Initialize context as a dictionary
        context = {
            "example": example,  # Original example
            "indexed_chunks": None,
            "retrieved_candidates": None,
            "grounded_candidates": None,
            "ranked_candidates": None,
            "top1_long": None,
            "short_answer": None,
            "short_answer_index": None,
            "cut_answer": None,
            "score": None
        }
        time_logger = TimeLogger()
        
        # Add indexed document
        time_logger.start('indexing')
        context["indexed_chunks"] = indexing.convert_to_indexed_format(
            context, distance=10, model_name="llama", max_tokens=300, overlap=60
        )
        time_logger.end('indexing')
        
        # Retrieve candidates
        time_logger.start('retrieving')
        context["retrieved_candidates"] = chunk_and_retrieve.retrieve(
            context, example=context["example"], verbose=False
        )
        time_logger.end('retrieving')
        
        # Ground the retrieved candidates
        time_logger.start('grounding')
        context["grounded_candidates"] = indexing.grounding(context)
        time_logger.end('grounding')
        
        # Rank the candidates
        time_logger.start('ranking')
        context["ranked_candidates"] = rank_v2.rank(context)
        time_logger.end('ranking')
        
        # Get the top1 long answer
        time_logger.start('finding long')
        context['top1_long'] = indexing.find_long(context)
        time_logger.end('finding long')
        
        # Refine the ranked candidates
        time_logger.start('refining')
        # If you want to add wrong answer, change this variable
        wrong_answer = ""
        context['short_answer'] = refine.refine(
            context['example']['question_text'], context['top1_long'], wrong_answer
        )
        time_logger.end('refining')

        time_logger.start('answer2index')
        context["short_answer_index"] = indexing.answer2index(context, verbose=False)
        time_logger.end('answer2index')
        context["score"] = 0.5

        # Show prediction stats
        print('===============Final Stats===================')
        print(f"Original question           : {context['example']['question_text']}")
        print(f"# of indexed chunks         : {len(context['indexed_chunks'])}")
        print(f"# of retrieved candidates   : {len(context['retrieved_candidates'])}")
        print(f"# of grounded candidates    : {len(context['grounded_candidates'])}")
        # short answer and cut answer will be printed in answer2index if verbose is True
        print(f"Top1 long answer            : {context['top1_long']}")
        print(f"Short answer                : {context['short_answer']}")
        print(f"cut answer                  : {context['cut_answer']}")
        print(f"**grounded truth**          : {get_short_answers(context['example'])}")
        print(f"Final begin index           : {context['final_index'][0]}, Final end index: {context['final_index'][1]}")

        time_logger.show_time()
        
        return context["short_answer_index"], context["score"], time_logger.get_log()
    
    
    

    


