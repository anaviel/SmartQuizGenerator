import spacy
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import T5ForConditionalGeneration, T5Tokenizer
from llama_cpp import Llama

from quiz_gen_django.gen_abc import QuestionGeneratorABC
from quiz_gen_django import model_gen_qa
from quiz_gen_django import model_gen_incorrect


def _load_ruT5_qa_models(
    question_model_path: str,
    answer_model_path: str
) -> tuple[PreTrainedModel, PreTrainedModel]:
    # модель для генерации вопросов
    question_model = T5ForConditionalGeneration.from_pretrained(question_model_path)
    # модель для генерации ответов
    answer_model = T5ForConditionalGeneration.from_pretrained(answer_model_path)
    return question_model, answer_model


class LocalQuestionGenerator(QuestionGeneratorABC):
    def __init__(
        self,
        questions_model_path: str,
        answers_model_path: str,
        incorrect_answers_model_path: str
    ):
        # загрузка модели для Русского языка - для nlp-обработки текста
        # модель (SpaCy):
        #   - https://huggingface.co/spacy/ru_core_news_sm
        self.nlp_pipeline = spacy.load("ru_core_news_sm")
        
        # tokenizer для моделей основанных на ruT5
        # -> взят от базовой модели:
        #   - https://huggingface.co/ai-forever/ruT5-base
        self.tokenizer_T5: PreTrainedTokenizerBase = T5Tokenizer.from_pretrained(
            "ai-forever/ruT5-base"
        )

        # инициализация модели для эмбеддингов
        self.embedder = SentenceTransformer(
            "distiluse-base-multilingual-cased-v1"
        )
        
        q, a = _load_ruT5_qa_models(
            questions_model_path,
            answers_model_path
        )
        
        self.question_model = q
        self.answer_model = a
        
        # инициализация модели
        self.saiga_model_gguf = Llama(
            model_path=incorrect_answers_model_path,
            n_ctx=2000, # максимальный контекст
            n_parts=1   # число частей модели (1 для GGUF)
        )
    
    def generate_questions(
        self,
        context: str
    ) -> tuple[str, list[str]]:
        return model_gen_qa.generate_questions(
            nlp=self.nlp_pipeline,
            embedder=self.embedder,
            tokenizer=self.tokenizer_T5,
            question_model=self.question_model,
            text=context
        )

    def generate_correct_answer(
        self,
        context: str,
        question: str
    ) -> str:
        return model_gen_qa.generate_correct_answers(
            tokenizer=self.tokenizer_T5,
            answer_model=self.answer_model,
            context=context,
            question=question
        )

    def generate_incorrect_answers(
        self,
        context: str,
        question: str,
        correct_answer: str
    ) -> list[str]:
        return model_gen_incorrect.generate_incorrect_answers(
            model=self.saiga_model_gguf,
            context=context,
            question=question,
            correct_answer=correct_answer
        )
