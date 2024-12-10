import openai
from typing import Dict, List, Optional, Tuple
import datetime
import json
from natural_questions import text_utils
from abc import ABC, abstractmethod
import tiktoken
import time



def strip_end_punctuation(text):
    # Common punctuation marks to remove from the end
    punctuation = '.!?,;:)"'
    
    # Strip whitespace first, then remove punctuation from the end
    text = text.strip()
    while text and text[-1] in punctuation:
        text = text[:-1].strip()
    
    return text

def get_nq_tokens(simplified_nq_example):
  """Returns list of blank separated tokens."""

  if "document_text" not in simplified_nq_example:
    raise ValueError("`get_nq_tokens` should be called on a simplified NQ"
                     "example that contains the `document_text` field.")

  return simplified_nq_example["document_text"].split(" ")

def get_short_answers(nq_example):
    document_tokens = get_nq_tokens(nq_example)
    short_answers = []
    for annotation in nq_example['annotations']:
        if annotation['short_answers']:
            for short_answer in annotation['short_answers']:
                short_answer_text = ' '.join(
                    document_tokens[
                        short_answer['start_token']:short_answer['end_token']]
                    )
                short_answers.append(short_answer_text)
    return short_answers


class TimeLogger:
    def __init__(self):
        self.start_time = time.time()
        self.time_log = {}
        self.start_log = {}

    def start(self, message):
        self.start_log[message] = time.time()

    def end(self, message):
        self.time_log[message] = time.time() - self.start_log.get(message, self.start_time)

    def show_time(self, message=None):
        if message is None:
            print('All time: =============================')
            for message in self.time_log:
                print(f"{message} time: {self.time_log[message]}s")
            print('============================')
        elif message in self.time_log:
            print(f"{message} time: {self.time_log[message]}s")
        elif message in self.start_log:
            print(f"{message} time continuing: {time.time() - self.start_log[message]}s")
        else:
            print(f"{message} not found")

    def get_log(self):
        return self.time_log


class BaseAgentSystem:
    def __init__(self):
        pass
    
    @abstractmethod
    def predict(self, example: Dict, verbose: bool = False) -> Tuple[str, float]:
        pass

    def _find_seq_index(self, document_text: str, seq: str) -> Tuple[int, int]:
        text_index = document_text.find(seq)
        if text_index == -1:
            return -1, -1
        start_token = document_text[:text_index].count(' ')
        end_token = start_token + len(seq.split(' '))
        return start_token, end_token

    def format_prediction(self, example: Dict, prediction: List, score: float) -> Dict:
        """
        Format the prediction into the format of Natural Questions evaluation
        
        Args:
            example: Single Natural Questions example
            prediction: Predicted answer string
            score: Score of the prediction
            
        Returns:
            Prediction(dict) in the format of Natural Questions evaluation

        Prediction format:
        {'predictions': [
            {
            'example_id': -2226525965842375672,
            'long_answer': {
                'start_byte': 62657, 'end_byte': 64776,
                'start_token': 391, 'end_token': 604
            },
            'long_answer_score': 13.5,
            'short_answers': [
                {'start_byte': 64206, 'end_byte': 64280,
                'start_token': 555, 'end_token': 560}, ...],
            'short_answers_score': 26.4,
            'yes_no_answer': 'NONE'
            }, ... ]
        }
        """
        pred_str = ' '.join(example['document_text'].split(" ")[prediction[0]:prediction[1]])
        prediction_dict = {
            'example_id': example['example_id'],
            'long_answer': {'start_byte': -1, 'end_byte': -1, 'start_token': -1, 'end_token': -1},
            'long_answer_score': -1,
            'short_answers': [{'start_byte': -1, 'end_byte': -1, 'start_token': -2, 'end_token': -1}],
            'short_answers_score': -1,
            'yes_no_answer': 'NONE',
            'prediction': pred_str
        }
        
        prediction_dict['short_answers'][0]['start_token'] = prediction[0]
        prediction_dict['short_answers'][0]['end_token'] = prediction[1]
        prediction_dict['short_answers_score'] = score

        return prediction_dict
    
    def read_context_and_format(self, context_path: str = None, save_path: str = None) -> List[Dict]:
        if context_path is None:
            context_path = self.context_path
        predictions = {'predictions': []}
        with open(context_path, 'r') as f:
            for line in f:
                context = json.loads(line)
                pred_index, score = context["short_answer_index"], context["score"]
                pred_dict = self.format_prediction(context['example'], pred_index, score)
                predictions['predictions'].append(pred_dict)

        if save_path is None:
            save_path = f"predictions_{self.model}_{time_str}.jsonl"
        with open(save_path, 'w') as f:
            f.write(json.dumps(predictions) + '\n')

        return predictions

    def predict_batch(self, examples: List[Dict], context_path: Optional[str] = None, verbose: bool = False) -> List[str]:
        """
        Make predictions for a batch of examples.
        """
        predictions = {'predictions': []}
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.time_str = time_str
        if context_path is None:
            context_path = f"{self.model}_{time_str}_context.jsonl"
        self.context_path = context_path
        for i, raw_example in enumerate(examples):
            if verbose:
                print(f"\nExample {i+1}/{len(examples)}")
            if 'document_text' not in raw_example:
                print('simplifying...')
                example = text_utils.simplify_nq_example(raw_example)
                
            else:
                example = raw_example
            print('char length:', len(example['document_text']))
            print('token length:', len(get_nq_tokens(example)))
            context, time_log = self.predict(example, verbose)
            if context['vectorstore'] is not None:
                ids = context['vectorstore'].get()['ids']
                # print("len(ids): ", len(ids))
                # print(context['vectorstore'].get())
                context['vectorstore'].delete(ids=ids)
                context['vectorstore'] = None
            # save prediction context
            with open(context_path, 'a') as f:
                f.write(json.dumps(context) + '\n')
            time.sleep(60)



class NQMultiAgent(BaseAgentSystem):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the multi-agent system.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for predictions
        """
        super().__init__()
        if api_key is None:
            self.client = openai.OpenAI()
        else:
            self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def _extract_answer(self, nq_example: Dict, model: str = None) -> str:
        """
        First agent: Extract answer from context.
        """
        if model is None and self.model is None:
            llm_model = "gpt-4o-mini"
        elif llm_model is None:
            llm_model = self.model
        else:
            llm_model = model

        response = self.client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant \
that provides concise answers."},
                {"role": "user", "content": f"Question: \
{nq_example['question_text']}\nContext: {nq_example['document_text']}\n\
Provide a brief answer(The answer has to be exactly from the context, \
which means the answer is a substring of the content):"}
            ]
        )
        return response.choices[0].message.content
    
    def _cut_answer(self, question: str, initial_answer: str, model: str = None) -> str:
        """
        Second agent: Cut the answer to the shortest possible substring of the given answer that can answer the question.
        """
        if model is None and self.model is None:
            llm_model = "gpt-4o-mini"
        elif llm_model is None:
            llm_model = self.model
        else:
            llm_model = model

        response = self.client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a professional judge \
on the question and answer pair. Cut the answer to the shortest contiguous \
substring of the given answer that can answer the question."},
                {"role": "user", "content": f"Question: {question}\n\
Initial answer: {initial_answer}\nProvide the shortest contiguous substring \
of the given answer that can answer the question:"}
            ]
        )
        return response.choices[0].message.content
    
    def _refine_answer(self, question: str, prev_answer: str, model: str = None) -> str:
        """
        Third agent: Refine and shorten the answer.
        """
        if model is None and self.model is None:
            llm_model = "gpt-4o-mini"
        elif llm_model is None:
            llm_model = self.model
        else:
            llm_model = model

        response = self.client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a professional judge \
on the question and answer pair. Cut the answer to the shortest possible \
substring of the given answer that can answer the question."},
                {"role": "user", "content": f"Question: {question}\n\
Previous answer: {prev_answer}\nProvide the shortest substring of the given \
answer that can answer the question:"}
            ]
        )
        return response.choices[0].message.content

    def predict(self, example: Dict, verbose: bool = False) -> Tuple[str, float]:
        """
        Make prediction using the multi-agent system.
        
        Args:
            example: SingleNatural Questions example
            verbose: Whether to print intermediate steps
            
        Returns:
            predicted string, and its score
        """
        print(f'Question: {example["question_text"]}')
        # First agent extracts answer
        initial_answer = self._extract_answer(example)
        if verbose:
            print("Initial answer:", initial_answer)
            
        # Second agent cuts answer
        cut_answer = self._cut_answer(example['question_text'], initial_answer)
        if verbose:
            print("Cut answer:", cut_answer)
        refined_answer = strip_end_punctuation(cut_answer)
        # Third agent refines answer
        gpt_answer_prev = ''
        while refined_answer != gpt_answer_prev:
            gpt_answer_prev = refined_answer
            refined_answer = self._refine_answer(example['question_text'], cut_answer)
            if verbose:
                print("Refined answer:", refined_answer)

        score = 50
        if verbose:
            print('- '*10)
            short_answer = get_short_answers(example)
            print('gold_answer:', short_answer)
            print('final_answer:', refined_answer)
        return refined_answer, score
    