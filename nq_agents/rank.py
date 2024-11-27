from autogen import ConversableAgent

def create_rank_agent(llm_config):
    return ConversableAgent(
        name="RankAgent",
        system_message="""
You are an agent who ranks the given candidate answers according to whether they can directly answer the given question.
The output is the ranking of the candidates, the integer index of the candidates.

# Output Format
{
"reasoning of top 1": "Why it is the first place",
"Top 1": index of the first candidate,
"reasoning of top 2": "Why it is the second place",
"Top 2": index of the second candidate,
"reasoning of top 3": "Why it is the third place",
"Top 3": index of the third candidate,
...
}
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )


def rank(agent, extracted_contents, question):

    return {}