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

    def send_request(self, question, long_answer, temperature=0.0, max_tokens=512):
        """
        send request to Ollama API to generate short answer
        """
        # Define prompt 
        prompt_template = """
        You are an intelligent assistant. Your task is to read a provided text and identify a specific answer to the given question. The answer must be:
        1. As short as possible, ideally just a few words, should not be a sentence.
        2. Most important: Taken directly and exactly from the **Long Answer** (continuous words).
        3. Do not summarize or create new words, only locate and extract the answer.
        4. If there is no answer, **Short Answer** should be None
        5. Just give the answer itself
        
        Here are some examples to learn from:
        **Example 1**
        **Question**:
        When are hops added to the brewing process?
        **Long Answer**:
        After mashing, the beer wort is boiled with hops (and other flavourings if used) in a large tank known as a "copper" or brew kettle – though historically the mash vessel was used and is still in some small breweries. The boiling process is where chemical reactions take place, including sterilization of the wort to remove unwanted bacteria, releasing of hop flavours, bitterness and aroma compounds through isomerization, stopping of enzymatic processes, precipitation of proteins, and concentration of the wort. Finally, the vapours produced during the boil volatilise off-flavours, including dimethyl sulfide precursors. The boil is conducted so that it is even and intense – a continuous "rolling boil". The boil on average lasts between 45 and 90 minutes, depending on its intensity, the hop addition schedule, and volume of water the brewer expects to evaporate. At the end of the boil, solid particles in the hopped wort are separated out, usually in a vessel called a "whirlpool".
        **Short Answer**:
        The boiling process

        **Example 2**
        **Question**:
        Where is the world's largest ice sheet located today?
        **Long Answer**:
        The Antarctic ice sheet is the largest single mass of ice on Earth. It covers an area of almost 14 million km and contains 30 million km of ice. Around 90% of the Earth's ice mass is in Antarctica, which, if melted, would cause sea levels to rise by 58 meters. The continent-wide average surface temperature trend of Antarctica is positive and significant at > 0.05°C/decade since 1957.
        **Short Answer**:
        Antarctica
        
         **Example 3**
        **Question**:
        where does the last name hogan come from?
        **Long Answer**:
        Hogan is an Irish surname . If derived from the Irish Gaelic , Ó hÓgáin , it is diminutive of Og meaning " young " . If it is derived from Cornish , it means " mortal " . This youthful definition of the name is also reflected in the Welsh , where Hogyn means stripling . The word Hogen means high in Dutch .
        **Short Answer**:
        None
        
        **Example 4**
        **Question**:
        who lives in the imperial palace in tokyo?
        **Long Answer**:
        The Tokyo Imperial Palace ( 皇居 , Kōkyo , literally " Imperial Residence " ) is the primary residence of the Emperor of Japan . It is a large park - like area located in the Chiyoda ward of Tokyo and contains buildings including the main palace ( 宮殿 , Kyūden ) , the private residences of the Imperial Family , an archive , museums and administrative offices.    
        **Short Answer**:
        the Imperial Family
        
        **Example 5**
        **Question**:
        where is the bowling hall of fame located?
        **Long Answer**:
        The World Bowling Writers ( WBW ) International Bowling Hall of Fame was established in 1993 and is located in the International Bowling Museum and Hall of Fame , on the International Bowling Campus in Arlington , Texas.
        **Short Answer**:
        Arlington , Texas
        
        **Example 6**
        **Question**:
        who won the election for mayor of cleveland?
        **Long Answer**:
        The 2017 Cleveland mayoral election took place on November 7 , 2017 , to elect the Mayor of Cleveland , Ohio . The election was officially nonpartisan , with the top two candidates from the September 12 primary election advancing to the general election , regardless of party . Incumbent Democratic Mayor Frank G . Jackson won reelection to a fourth term.
        **Short Answer**:
        Incumbent Democratic Mayor Frank G . Jackson
        
        **Example 7**
        **Question**:
        when did the watts riot start and end?
        **Long Answer**:
        The Watts riots , sometimes referred to as the Watts Rebellion , took place in the Watts neighborhood of Los Angeles from August 11 to 16 , 1965 .
        **Short Answer**:
        August 11 to 16 , 1965
        
        **Example 8**
        **Question**:
        when did kendrick lamars first album come out?
        **Long Answer**:
        On the topic of whether his next project would be an album or a mixtape , Lamar answered : " I treat every project like it's an album anyway . It's not going to be nothing leftover . I never do nothing like that . These are my leftover songs you all can have them . I'm going to put my best out . My best effort . I'm trying to look for an album in 2012 . " In June 2011 , Lamar released " Ronald Reagan Era ( His Evils ) " , a cut from Section . 80 , featuring Wu - Tang Clan leader RZA . On July 2 , 2011 , Lamar released Section . 80 , his first independent album , to critical acclaim . The album features guest appearances from GLC , Colin Munroe , Schoolboy Q , and Ab - Soul , while the production was handled by Top Dawg in - house production team Digi + Phonics as well as Wyldfyer , Terrace Martin and J . Cole . Section . 80 went on to sell 5,300 digital copies in its first week , without any television or radio coverage and received mostly positive reviews .
        **Short Answer**:
        July 2 , 2011
        
        **Example 9**
        **Question**:
        where does the energy in a nuclear explosion come from?
        **Long Answer**:
        A nuclear explosion is an explosion that occurs as a result of the rapid release of energy from a high - speed nuclear reaction . The driving reaction may be nuclear fission , nuclear fusion or a multistage cascading combination of the two , though to date all fusion - based weapons have used a fission device to initiate fusion , and a pure fusion weapon remains a hypothetical device .
        **Short Answer**:
        high - speed nuclear reaction
        
        
        **Current Task**
        **Question**:
        {question}
        **Long Answer**:
        {long_answer}
        **Short Answer**:
        """

        # fill template
        prompt = prompt_template.format(question=question, long_answer=long_answer)
        
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

def refine(question: str, long_answer: str) -> str:
    agent = OllamaAgent()
    try:
        short_answer = agent.send_request(question, long_answer)
        return short_answer
    except Exception as e:
        return e

# Simple test with main
if __name__ == "__main__":
    question = "what is the widest highway in north america"
    long_answer = """
    King's Highway 401, commonly referred to as Highway 401 and also known by its official name as the Macdonald–Cartier Freeway or colloquially as the four-oh-one,[3] is a controlled-access400-series highway in the Canadian province of Ontario. It stretches 828.0 kilometres (514.5 mi) from Windsor in the west to the Ontario–Quebec border in the east. The part of Highway 401 that passes through Toronto is North America's busiest highway,[4][5] and one of the widest.[6][7] Together with Quebec Autoroute 20, it forms the road transportation backbone of the Quebec City–Windsor Corridor, along which over half of Canada's population resides and is also a Core Route in the National Highway System of Canada. The route is maintained by the Ministry of Transportation of Ontario (MTO) and patrolled by the Ontario Provincial Police. The speed limit is 100 km/h (62 mph) throughout its length, unless posted otherwise.
    """
    print("short answer:", refine(question, long_answer))





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