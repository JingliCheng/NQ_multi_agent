from autogen import ConversableAgent
from typing import Tuple


def create_refine_answer_agent(llm_config):
    return ConversableAgent(
        name="RefineAnswerAgent",
        system_message="""
You are an agent who refines the given candidate answers to answer the given question.
Output:
{
"refined_answer": "refined answer"
}
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

def refine(agent, ranked_candidates, question) -> Tuple[str, float]:
    return ('', 0.5)