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
from nq_agents.multi_agent import BaseAgentSystem

from nq_agents import indexing
from nq_agents import chunk_and_retrieve
from nq_agents import rank
from nq_agents import refine


# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "placeholder" # Replace with your actual API key

# QPS and concurrency limits
MAX_QPS = 2  # 每秒最多请求数
MAX_CONCURRENT_REQUESTS = 2  # 最大并发请求数

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
        max_qps=2,
        max_concurrent_requests=2,
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
            "indexed_answer": None,
            "score": None
        }
        
        # Add indexed document
        context["indexed_chunks"] = indexing.convert_to_indexed_format(
            context, distance=10, model_name="llama", max_tokens=1000, overlap=100
        )
        
        # Retrieve candidates
        context["retrieved_candidates"] = chunk_and_retrieve.retrieve(
            context, example=context["indexed_example"], verbose=True
        )
        
        # Ground the retrieved candidates
        context["grounded_candidates"] = indexing.grounding(
            context, context["retrieved_candidates"], context["example"]
        )
        
        # Rank the candidates
        context["ranked_candidates"] = rank.rank(
            context, context["indexed_example"], context["grounded_candidates"]
        )
        
        # Refine the ranked candidates
        answer, score = refine.refine(
            context, context["indexed_example"], context["ranked_candidates"]
        )

        context["indexed_answer"] = indexing.answer2index(answer)
        context["score"] = score
        
        return context["indexed_answer"], context["score"]
    
    
    

    


