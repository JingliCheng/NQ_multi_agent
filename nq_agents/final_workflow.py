import json
from nq_agents.multi_agent import BaseAgentSystem
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
from autogen import GroupChatManager
from autogen import ConversableAgent
from autogen import UserProxyAgent
from autogen import AssistantAgent
from autogen import GroupChat
import autogen as autogen

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "placeholder" # Replace with your actual API key
os.environ["AUTOGEN_USE_DOCKER"] = "False"

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

# Maximum attempts for the final workflow
MAX_ATTEMPTS = 3

class FinalWorkflowAutogen(BaseAgentSystem):
    def __init__(
        self,
        llm_provider="ollama",
        api_key=None,
        max_tokens=20480,
        max_qps=15,
        max_concurrent_requests=15,
    ):
        super().__init__()
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.max_qps = max_qps
        self.max_concurrent_requests = max_concurrent_requests
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

    def create_validation_agents(self):
        from autogen import ConversableAgent
        prompts = [
            """
            # Role

                You are a Validation Agent focusing on ensuring factual correctness in answers. Your primary task is to determine if the provided answer aligns accurately with the information in the given context.

                # Instructions

                1. **Strict Analysis**: Carefully verify whether the answer is factually supported by the provided context.
                2. **Reasoning**: Provide a concise reasoning process explaining why the answer is factually correct or incorrect.
                3. **Formatting**: Your output must strictly follow the specified format and contain only JSON. Avoid including any other text or explanation outside the JSON format.

                # Input Format
                [Question] Question Text  
                [Answer] Proposed Answer  
                [Context] Context Text  

                # Output Format
                {
                    "vote": true/false,
                    "reasoning": "Explain your reasoning."
                }

                # Examples

                ## Example 1: Correct Answer

                **Input:**  
                [Question] What is the capital city of France?  
                [Answer] Paris  
                [Context] Paris is the capital city of France.

                **Output:**  
                {
                    "vote": true,
                    "reasoning": "The answer correctly states that Paris is the capital city of France as supported by the context."
                }

                ## Example 2: Incorrect Answer

                **Input:**  
                [Question] What is the capital city of France?  
                [Answer] London  
                [Context] Paris is the capital city of France.

                **Output:**  
                {
                    "vote": false,
                    "reasoning": "The answer incorrectly states London as the capital city of France, while the context confirms it is Paris."
                }
            """,
            """
            # Role

                You are a Validation Agent focusing on ensuring factual correctness in answers. Your primary task is to determine if the provided answer aligns accurately with the information in the given context.

                # Instructions

                1. **Strict Analysis**: Carefully verify whether the answer is factually supported by the provided context.
                2. **Reasoning**: Provide a concise reasoning process explaining why the answer is factually correct or incorrect.
                3. **Formatting**: Your output must strictly follow the specified format and contain only JSON. Avoid including any other text or explanation outside the JSON format.

                # Input Format
                [Question] Question Text  
                [Answer] Proposed Answer  
                [Context] Context Text  

                # Output Format
                {
                    "vote": true/false,
                    "reasoning": "Explain your reasoning."
                }

                # Examples

                ## Example 1: Correct Answer

                **Input:**  
                [Question] What is the capital city of France?  
                [Answer] Paris  
                [Context] Paris is the capital city of France.

                **Output:**  
                {
                    "vote": true,
                    "reasoning": "The answer correctly states that Paris is the capital city of France as supported by the context."
                }

                ## Example 2: Incorrect Answer

                **Input:**  
                [Question] What is the capital city of France?  
                [Answer] London  
                [Context] Paris is the capital city of France.

                **Output:**  
                {
                    "vote": false,
                    "reasoning": "The answer incorrectly states London as the capital city of France, while the context confirms it is Paris."
                }
            """,
            """
            # Role

                You are a Validation Agent focusing on ensuring factual correctness in answers. Your primary task is to determine if the provided answer aligns accurately with the information in the given context.

                # Instructions

                1. **Strict Analysis**: Carefully verify whether the answer is factually supported by the provided context.
                2. **Reasoning**: Provide a concise reasoning process explaining why the answer is factually correct or incorrect.
                3. **Formatting**: Your output must strictly follow the specified format and contain only JSON. Avoid including any other text or explanation outside the JSON format.

                # Input Format
                [Question] Question Text  
                [Answer] Proposed Answer  
                [Context] Context Text  

                # Output Format
                {
                    "vote": true/false,
                    "reasoning": "Explain your reasoning."
                }

                # Examples

                ## Example 1: Correct Answer

                **Input:**  
                [Question] What is the capital city of France?  
                [Answer] Paris  
                [Context] Paris is the capital city of France.

                **Output:**  
                {
                    "vote": true,
                    "reasoning": "The answer correctly states that Paris is the capital city of France as supported by the context."
                }

                ## Example 2: Incorrect Answer

                **Input:**  
                [Question] What is the capital city of France?  
                [Answer] London  
                [Context] Paris is the capital city of France.

                **Output:**  
                {
                    "vote": false,
                    "reasoning": "The answer incorrectly states London as the capital city of France, while the context confirms it is Paris."
                }
            """
        ]

        agents = []
        for i, prompt in enumerate(prompts):
            agents.append(
                ConversableAgent(
                    name=f"ValidationAgent_{i+1}",
                    system_message=prompt,
                    llm_config=self.llm_config,
                    human_input_mode="NEVER",
                )
            )
        return agents

    def create_judge_agent(self):
        from autogen import ConversableAgent
        return ConversableAgent(
            name="JudgeAgent",
            system_message="""
                You are a Judge Agent.
                Input: { "validation_results": [...], "proposed_answer": "..."}

                If all validation_results.vote == true, return:
                {"consensus": True, "final_answer": "<proposed_answer>", "negative_feedback": ""}

                If any vote == False, return:
                {"consensus": False, "final_answer": null, "negative_feedback": "Please refine the answer."}
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
    
    def call_validation_agents(self, agents, question, answer, context_text):
        results = []
        for agent in agents:
            input_message = f"Question: {question}\nAnswer: {answer}\nContext: {context_text}"
            print("input_message:", input_message)
            response = agent.generate_reply(messages=[{"content": input_message, "role": "user"}])
            print("validation response:" + response)
            print(f"Agent {agent.name} response: {response}")
            try:
                data = json.loads(response)
                if "vote" in data and "reasoning" in data:
                    results.append(data)
                else:
                    results.append({"vote": False, "reasoning": "Invalid JSON format"})
            except:
                results.append({"vote": False, "reasoning": "JSON parse error"})
        return results

    def call_judge_agent(self, agent, validation_results, proposed_answer):
        inp = json.dumps({"validation_results": validation_results, "proposed_answer": proposed_answer})
        print("judge input:", inp)
        response = agent.generate_reply(messages=[{"content": inp, "role": "user"}])
        content = response
        print("judge response:", content)
        try:
            data = json.loads(content)
            return data
        except:
            return {"consensus": False, "final_answer": None, "negative_feedback": "Judge parse error"}

    def predict_answer(self, example: Dict, verbose: bool = False, nagativeFeedback: str = "") -> Dict[str, any]:
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
        return context
        
    def predict(self, example: Dict, verbose: bool = False) -> Tuple[str, float]:

        """
        new running process:
        1. Predict and get answer
        2. ValidationAgents
        3. Judgement
        4. 若JudgeAgent否决，传递negative_feedback（此处仅打印或留待下次迭代中模拟使用）并重复
        """

        
        # 新增的循环逻辑
        validation_agents = self.create_validation_agents()
        judge_agent = self.create_judge_agent()

        final_answer = None
        score = 0.0
        log = {}

        negative_feedback = []
        for attempt in range(1, MAX_ATTEMPTS+1):
            if verbose:
                print(f"Attempt {attempt}/{MAX_ATTEMPTS}")

            context = self.predict_answer(example, verbose=verbose, nagativeFeedback=negative_feedback)
            score = context["score"]
            proposed_answer = context["short_answer"]
            if verbose:
                print("Proposed Answer:", proposed_answer)

            # Validation
            results = self.call_validation_agents(validation_agents, example["question_text"], proposed_answer, context["top1_long"])
            print("Validation Results:", results)
            if verbose:
                print("Validation Results:", results)

            # Judge
            judge_result = self.call_judge_agent(judge_agent, results, proposed_answer)
            if verbose:
                print("Judge Result:", judge_result)

            if judge_result.get("consensus", False) is True:
                final_answer = judge_result["final_answer"]
                break
            else:
                # 没有达成共识，获取negative_feedback
                negative_feedback.append(judge_result.get("negative_feedback", "No consensus."))
                if verbose:
                    print("No consensus. Negative feedback:", negative_feedback)
                # negative_feedback 可以在下次调用 predict_once 时作为参考(当前代码未实际使用这个变量修改逻辑)
        
        if final_answer is None:
            # use the last answer
            final_answer = proposed_answer
        print(f"Final Answer: {final_answer}")
        # 返回最终结果
        return final_answer, score, log
    

    def predictv2(self, example: Dict, verbose: bool = False) -> Tuple[str, float]:

        """
        new running process:
        1. Predict and get answer
        2. ValidationAgents
        3. Judgement
        4. 若JudgeAgent否决，传递negative_feedback（此处仅打印或留待下次迭代中模拟使用）并重复
        """
        negative_feedback = []
        context = self.predict_answer(example, verbose=verbose, nagativeFeedback=negative_feedback)
        question = example["question_text"]
        long_answer = context["top1_long"]
        short_answer = context["short_answer"]
        
        ask_agent = self.create_conversational_ask_agent()
        validation_agent = self.create_conversational_validation_agent()
        judge_agent = self.create_conversational_judge_agent()

        final_answer = None
        score = 0.0
        log = {}

        user_message = [
            {
                "role": "user",
                "content": f"""
                Question: {question}
                Long Answer: {long_answer}
                Short Answer: {short_answer}
                Please validate if this short answer is correct based on the long answer.
                """
            }
        ]
        # Construct the initial message
        initial_message =f"""
        Question: {question}
        Long Answer: {long_answer}
        Short Answer: {short_answer}
        
        Please validate if this short answer is correct based on the long answer.
        """
        if (verbose):
            print("initial_message:", initial_message)
        
        
        for attempt in range(1, MAX_ATTEMPTS+1):
            if verbose:
                print(f"Attempt {attempt}/{MAX_ATTEMPTS}")
                print(f"AskAgent - Question: {question}")
                print(f"Long Answer: {long_answer}")
                print(f"Proposed Short Answer: {short_answer}")
                print("start chat")
            # 初始化 AskAgent 对 ValidationAgent 的对话
            # ask_agent.initiate_chat(
            #     validation_agent,
            #     message=f"""
            #     Question: {question}
            #     Long Answer: {long_answer}
            #     Short Answer: {short_answer}
            #     Request: Please validate whether the short answer is correctly derived 
            #     from the long answer.
            #     """,
            # )

             # Start conversation via JudgeAgent

            if short_answer is None:
                short_answer = "NO_ANSWER_PROVIDED"


            try:
                # Start the group chat with the initial message
                chat_result = ask_agent.initiate_chat(
                    recipient = validation_agent,
                    messages= [user_message],
                    max_turns=3,
                    verbose=verbose,
                )
                print("chat_result:", chat_result)
                # Parse validation result
                try:
                    validation_result = json.loads(chat_result)
                    if validation_result.get("decision", False):
                        final_answer = short_answer
                    else:
                        final_answer = validation_result.get("suggested_answer", short_answer)
                        
                    score = 1.0 if validation_result.get("decision", False) else 0.5
                    
                except json.JSONDecodeError:
                    final_answer = short_answer
                    score = 0.5
                    
            except Exception as e:
                print(f"Error during chat: {e}")
                print(f"Chat error: {str(e)}")
                print(f"Ask agent config: {ask_agent.llm_config}")
                print(f"Validation agent config: {validation_agent.llm_config}")
                final_answer = short_answer
                score = 0.0

        return final_answer, score, {"chat_result": chat_result}

    
    def create_conversational_ask_agent(self):

        ask_agent = ConversableAgent(
            name="AskAgent",
            llm_config=self.llm_config,
            system_message="""
            Now you are a User. Your task is to ask questions to the ValidationAgent
            to clarify the short answer and ensure its correctness. 
            If the ValidationAgent provides a final decision in JSON format with 'decision' 
            (true/false), do not ask further questions. End the conversation.
            """,
            human_input_mode="NEVER",
        )
        return ask_agent

    def create_conversational_validation_agent(self):

        validation_agent = ConversableAgent(
        name="ValidationAgent",
        llm_config=self.llm_config,
        system_message="""
            You are ValidationAgent. Your task is to validate whether the short 
            answer provided by AskAgent is correct using the long answer as evidence.
            Instructions:
            1. **Analysis**: Validate the short answer based on the long answer.
            2. **Discussion**: Engage in at most two discussion rounds with AskAgent.
            3. **Decision**: Provide a final decision in JSON format with 'decision' (true/false)
               and a concise 'reasoning' for the decision. Example:
               {"decision": true, "reasoning": "Short answer is factually correct."}
            4. You must Discuss first before making a decision. 
            Ensure strict JSON formatting. Do not include any non-JSON text..
                
                If their answer is None, you should suggest a better short answer. Short answer should typically be less than 3 words.
            """,
        human_input_mode="NEVER",
        )
        return validation_agent

    def create_conversational_judge_agent(self):
        return ConversableAgent(
            name="JudgeAgent",
            llm_config=self.llm_config,
            system_message="""You are a Judge analyzing conversation history between AskAgent and ValidationAgent.
            Analyze their conversation and determine if they reached consensus about the answer.
            Return a JSON response in this format:
            {
                "consensus": true/false,
                "final_answer": "the agreed answer or null",
                "reasoning": "explanation of your decision"
            }""",
            human_input_mode="NEVER",
        )