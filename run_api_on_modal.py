import os
import random

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse
import modal
from modal import App, Image, gpu, enter, method, asgi_app
from pathlib import Path

import spacy
from typing import Sequence
from sentence_transformers import SentenceTransformer, util
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import T5ForConditionalGeneration, T5Tokenizer

import re
from llama_cpp import Llama

import wikipedia
# установка языка для Wikipedia - Русский
wikipedia.set_lang("ru")


def wikipedia_search(keywords: list[str]):
    """
    Получает краткое описание из Википедии по ключевым словам, переданным в запросе.
    Если ключевые слова не найдены, возвращается сообщение об ошибке.
    """

    # объединяем ключевые слова в строку для поиска
    search_query = " ".join(keywords)
    print(f"Ищем информацию по запросу: {search_query}")

    # поиск статей по запросу
    try:
        # ограничиваем результаты поиска до 1 статьи по запросу
        search_results = wikipedia.search(search_query, results=1)
        print(f"Результаты поиска: {search_results}")

        if search_results:
            # получение заголовка 1 найденной страницы (статьи) подходящей под запрос
            page_title = search_results[0]
            # загрузка текста 1 найденной страницы (статьи) - по заголовку
            page = wikipedia.page(page_title)
            # возвращаем краткое описание страницы (статьи)
            return page.summary
        else:
            return "Информация по запросу не найдена."
    except wikipedia.exceptions.DisambiguationError as err:
        # TODO: запись ошибки в лог
        print(f"[WARN] при поиске '{keywords}' по Wiki возникла ошибка: {err}")
        # В случае неоднозначности
        return "Запрос слишком неоднозначен."


def generate_correct_answer(
    tokenizer: PreTrainedTokenizerBase,
    answer_model: PreTrainedModel,
    context: str,
    question: str
) -> str:
    """
    Генерирует ответы на переданные вопросы с использованием контекста.
    """
    # формируем запрос для модели: вопрос + контекст
    input_text = (
        question + " " + context
    )
    
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
    ).input_ids
    
    answer_ids: Sequence = answer_model.generate(
        input_ids,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    # декодируем ответ в текст
    correct_answer: str = tokenizer.decode(
        answer_ids[0], skip_special_tokens=True
    )

    # ответ будет начинаться с большой буквы
    correct_answer: str = correct_answer.capitalize()
    return correct_answer


def generate_questions(
    nlp: spacy.language.Language,
    embedder: SentenceTransformer,
    tokenizer: PreTrainedTokenizerBase,
    question_model: PreTrainedModel,
    text: str
) -> tuple[str, list[str]]:
    """
    Генерирует вопросы на основе переданного текста.

    Аргументы:
    text -- текст, на основе которого модель генерирует вопросы

    Возвращает:
    Список сгенерированных вопросов.
    """
    _similarity_threshold = 0.7
    _unique_q_limit = 3
    
    # обрабатываем текст с помощью модели SpaCy
    nlp_text = nlp(text.lower())

    # извлекаем ключевые слова
    # (существительные, прилагательные, глаголы) из текста
    keywords = [
        token.text
        for token in nlp_text
            if token.pos_ in ["NOUN", "ADJ", "VERB"]
    ]

    # получаем краткую информацию по ключевым словам из Википедии
    wiki_summary = (
        wikipedia_search(keywords) if keywords else "Нет ключевых слов для поиска."
    )

    if (
        (wiki_summary == "Нет ключевых слов для поиска.")
        or (wiki_summary == "Информация по запросу не найдена.")
        or (wiki_summary == "Запрос слишком неоднозначен.")
    ):
        question_text = None
        return wiki_summary, question_text

    # печатаем результат из Википедии для проверки
    print("Текст из Википедии:", wiki_summary)

    # генерация вопросов с использованием обученной модели для вопросов
    input_ids = tokenizer(
        wiki_summary,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
    ).input_ids

    question_ids = question_model.generate(
        input_ids,
        max_length=64,
        num_beams=12,
        num_return_sequences=12,    # Генерация 12 различных вопросов
        temperature=1.2,            # Используем более высокую температуру для разнообразия
        top_p=0.9,                  # Используем отбор по вероятности
        early_stopping=True,        # Останавливаем генерацию, когда модель "уверена" в результате
    )

    # декодируем вопросы в текст
    questions: list[str] = [
        tokenizer.decode(
            q, skip_special_tokens=True
        ) for q in question_ids
    ]

    # преобразуем вопросы в эмбеддинг-тензоры
    question_embeddings = embedder.encode(
        questions,
        convert_to_tensor=True
    )

    unique_questions = []
    for i, question in enumerate(questions):
        # сравниваем с ранее добавленными вопросами
        is_unique = True
        for unique_question in unique_questions:
            # вычисляем cos-схожесть между вопросами
            sim_score = util.pytorch_cos_sim(
                question_embeddings[i],
                unique_question[1]
            )
            # порог схожести
            if sim_score >= _similarity_threshold:
                is_unique = False
                break

        if is_unique and len(unique_questions) < _unique_q_limit:
            unique_questions.append(
                (question, question_embeddings[i])
            )
    # возвращаем только вопросы (без векторов)
    return wiki_summary, [q[0] for q in unique_questions]


# задаём параметры для модели неправильных ответов
SYSTEM_PROMPT       = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN        = 1587
USER_TOKEN          = 2188
BOT_TOKEN           = 12435
LINE_BREAK_TOKEN    = 13

# словарь токенов для ролей
ROLE_TOKENS = {
    "user":     USER_TOKEN,
    "bot":      BOT_TOKEN,
    "system":   SYSTEM_TOKEN
}


def get_message_tokens(
    model: Llama,
    role: str,
    content: str
) -> list[int]:
    """
    Преобразует текст сообщения в токены с добавлением специальных токенов.
    """
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINE_BREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(
    model: Llama
) -> list[int]:
    """
    Осуществляет вставку токенов для сообщения.
    """
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(
        model,
        **system_message
    )


def generate_incorrect_answers(
    model: Llama,
    context: str,
    question: str,
    correct_answer: str
) -> list[str]:
    """
    Осуществляет генерацию неправильных ответов с помощью модели: 'saiga_mistral_7b_gguf'
    """
    print(
        f"generate_incorrect_answers: context={context} ",
        f"correct_answer={correct_answer}"
    )

    # специальные токены
    system_tokens: list[int] = get_system_tokens(
        model
    )

    # формируем промт
    prompt = (
        "Задание: придумай 3 осмысленных, но НЕПРАВИЛЬНЫХ вариантов ответа "
        f"к вопросу: {question}, опираясь на правильный ответ: {correct_answer}."
    )


    # преобразуем промт в токены
    user_tokens = get_message_tokens(model, "user", prompt)
    role_tokens = [model.token_bos(), BOT_TOKEN, LINE_BREAK_TOKEN]
    tokens = system_tokens + user_tokens + role_tokens

    # генерация ответа
    generator = model.generate(
        tokens,
        top_k=30,
        top_p=0.9,
        temp=0.5,
        repeat_penalty=1.1
    )

    # сбор ответа
    response = ""
    for token in generator:
        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        if token == model.token_eos():
            break
        response += token_str

    # разбиваем ответ на список неправильных вариантов
    incorrect_answers = [
        ans.strip()
        for ans in response.split("\n")
            if ans.strip()
    ]

    # убираем нумерацию (например, "1. ", "2. ", "3. ")
    # с помощью регулярного выражения
    incorrect_answers = re.sub(
        r"^\d+\.\s*", "",
        response,
        flags=re.MULTILINE
    ).strip().split("\n")[:3]

    # убираем лишние пустые строки, если они есть
    incorrect_answers: list[str] = [
        answer.strip()
        for answer in incorrect_answers
            if answer.strip()
    ]
    
    print(f"Неправильные ответы: {incorrect_answers}")
    return incorrect_answers


def _load_ruT5_qa_models(
    question_model_path: str,
    answer_model_path: str
) -> tuple[PreTrainedModel, PreTrainedModel]:
    # модель для генерации вопросов
    question_model = T5ForConditionalGeneration.from_pretrained(question_model_path)
    # модель для генерации ответов
    answer_model = T5ForConditionalGeneration.from_pretrained(answer_model_path)
    return question_model, answer_model


VOLUME_WITH_MODELS: str = "ruT5_quiz_gen"

QUESTIONS_MODEL_PATH            = os.getenv("QUESTIONS_MODEL_PATH")
ANSWERS_MODEL_PATH              = os.getenv("ANSWERS_MODEL_PATH")
INCORRECT_ANSWERS_MODEL_PATH    = os.getenv("INCORRECT_ANSWERS_MODEL_PATH")

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "spacy",
        "sentence-transformers",
        "sentencepiece",
        "transformers",
        "fastapi",
        "llama-cpp-python",
        "wikipedia"
        # "sse_starlette",
    ).apt_install(
        "wget"
    )
    .run_commands(
        "python -m spacy download ru_core_news_sm"
    )
)

app = App("ruT5_quiz_v3", image=image)
volume_with_models = modal.Volume.from_name(
    VOLUME_WITH_MODELS,
    create_if_missing=True
)

MODEL_DIR = Path("/models")


@app.cls(cpu=2, gpu=gpu.T4(count=1), volumes={MODEL_DIR: volume_with_models})
class ModelQandA:
    @enter()
    def init_models(self):
        print(MODEL_DIR, MODEL_DIR.exists())
        
        # печатаем текущую директорию
        current_dir = os.getcwd()
        print(f"Текущая директория: {current_dir}")

        # что-то вроде ls -la
        print("\nСодержимое директории (с подробностями):")
        for entry in os.scandir(MODEL_DIR):
            info = f"{'<d>' if entry.is_dir() else '<f>'} {entry.name}"
            print(info)
        
        questions_model_path: Path          = MODEL_DIR / "ruT5_quiz_gen_questions"
        answers_model_path: Path            = MODEL_DIR / "ruT5_quiz_gen_answers"
        incorrect_answers_model_path: Path  = MODEL_DIR / "saiga_mistral_7b_gguf" / "model-q4_K.gguf"
        
        print("[Model loading path]", str(questions_model_path))
        print("[Model loading path]", str(answers_model_path))
        print("[Model loading path]", str(incorrect_answers_model_path))
        
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
        
        self.question_model, self.answer_model = _load_ruT5_qa_models(
            str(questions_model_path),
            str(answers_model_path)
        )
        
        # инициализация модели
        self.saiga_model_gguf = Llama(
            model_path=str(incorrect_answers_model_path),
            n_ctx=2000, # максимальный контекст
            n_parts=1   # число частей модели (1 для GGUF)
        )

    @method()
    def gen_questions_(self, context: str) -> tuple[str, list[str]]:
        return generate_questions(
            nlp=self.nlp_pipeline,
            embedder=self.embedder,
            tokenizer=self.tokenizer_T5,
            question_model=self.question_model,
            text=context,
        )

    @method()
    def gen_one_correct_answer_(
        self,
        context: str,
        question: str
    ) -> str:
        return generate_correct_answer(
            tokenizer=self.tokenizer_T5,
            answer_model=self.answer_model,
            context=context,
            question=question,
        )

    @method()
    def gen_incorrect_answers_(
        self,
        context: str,
        question: str,
        correct_answer: str
    ) -> list[str]:
        return generate_incorrect_answers(
            model=self.saiga_model_gguf,
            context=context,
            question=question,
            correct_answer=correct_answer,
        )
    
    @method()
    def result_QA_generation(self, topic: str) -> list[dict] | str:
        """
        Основная функция, которая генерирует вопросы и ответы для переданного контекста.
        """
        # генерация вопросов
        wiki_summary, questions = generate_questions(
            nlp=self.nlp_pipeline,
            embedder=self.embedder,
            tokenizer=self.tokenizer_T5,
            question_model=self.question_model,
            text=topic
        )

        # TODO: проверить что этот кейс обрабатывается правильно
        if questions is None:
            return wiki_summary

        # генерация ответов
        final_results_qa: list[dict] = []
        
        for question in questions:
            correct_answer = generate_correct_answer(
                tokenizer=self.tokenizer_T5,
                answer_model=self.answer_model,
                context=wiki_summary,
                question=question
            )

            incorrect_answers = generate_incorrect_answers(
                self.saiga_model_gguf,
                context=wiki_summary,
                question=question,
                correct_answer=correct_answer
            )

            all_answers = [correct_answer] + incorrect_answers
            random.shuffle(all_answers)
            final_results_qa.append({
                "question": question,
                "answers": all_answers,
                "correct_answer": correct_answer,
            })

        print(f"[INFO] Сгенерированные вопросы и ответы: {final_results_qa}")
        return final_results_qa

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.function(image=image, volumes={MODEL_DIR: volume_with_models})
@web_app.get("/generate/generate_quiz")
async def generate_quiz(topic: str = Query(...)):    
    question_answers: list[dict] | str = ModelQandA().result_QA_generation.remote(
        topic
    )
    
    print(question_answers)
    
    if isinstance(question_answers, str):
        # если вопрос-ответ вернулся в виде строки (ошибка генерации)
        return JSONResponse(
            status_code=400,
            content={
                "error": "Failed to generate questions",
                "details": question_answers
            }
        )

    elif isinstance(question_answers, list):
        # если вопрос-ответ вернулся в виде списка (успешно сгенерировано)
        return {
            "topic": topic,
            "question_answers": question_answers
        }

    # обработка неожиданных ситуаций
    raise HTTPException(
        status_code=500,
        detail="Unexpected error occurred."
    )


@app.function(image=image, volumes={MODEL_DIR: volume_with_models})
@web_app.get("/generate/generate_questions")
async def generate_questions_(topic: str = Query(...)): 
    # генерация вопросов
    wiki_summary, questions = ModelQandA().gen_questions_.remote(topic)

    # TODO: проверить что этот кейс обрабатывается правильно
    if questions is None or len(questions) == 0:
        # если вопрос-ответ вернулся в виде строки (ошибка генерации)
        return JSONResponse(
            status_code=400,
            content={
                "error": "Failed to generate questions",
                "details": wiki_summary
            }
        )

    if len(questions):
        # если вопрос-ответ вернулся в виде списка (успешно сгенерировано)
        return {
            "topic": topic,
            "wiki_summary": wiki_summary,
            "questions": questions
        }

    # обработка неожиданных ситуаций
    raise HTTPException(
        status_code=500,
        detail="Unexpected error occurred."
    )


@app.function(image=image, volumes={MODEL_DIR: volume_with_models})
@web_app.get("/generate/generate_correct_answer")
async def generate_answer_correct_(
    context: str = Query(...),
    question: str = Query(...)
):
    # генерация ответа
    the_answer = ModelQandA().gen_one_correct_answer_.remote(
        context,
        question
    )

    if the_answer:
        # успешно сгенерировано
        return {
            "context": context,
            "question": question,
            "answer_correct": the_answer
        }

    # обработка неожиданных ситуаций
    raise HTTPException(
        status_code=500,
        detail="Unexpected error occurred."
    )


@app.function(image=image, volumes={MODEL_DIR: volume_with_models})
@web_app.get("/generate/generate_incorrect_answers")
async def generate_answers_incorrect(
    context: str = Query(...),
    question: str = Query(...),
    correct_answer: str = Query(...)
):
    # генерация
    all_answers = ModelQandA().gen_incorrect_answers_.remote(
        context,
        question,
        correct_answer
    )

    if all_answers:
        # успешно сгенерировано
        return {
            "context": context,
            "question": question,
            "correct_answer": correct_answer,
            "all_answers": all_answers
        }

    # обработка неожиданных ситуаций
    raise HTTPException(
        status_code=500,
        detail="Unexpected error occurred."
    )


@app.local_entrypoint()
def main():
    for model_directory in [
        QUESTIONS_MODEL_PATH,
        ANSWERS_MODEL_PATH,
        INCORRECT_ANSWERS_MODEL_PATH
    ]:
        if not Path(model_directory).is_dir():
            raise RuntimeError(f"Invalid path provided (directory expected): {model_directory}")
    
    with volume_with_models.batch_upload() as upload:
        upload: modal.volume.VolumeUploadContextManager
        upload.put_directory(QUESTIONS_MODEL_PATH,          "/ruT5_quiz_gen_questions")
        print(f"finished {QUESTIONS_MODEL_PATH}")
        upload.put_directory(ANSWERS_MODEL_PATH,            "/ruT5_quiz_gen_answers")
        print(f"finished {ANSWERS_MODEL_PATH}")
        upload.put_directory(INCORRECT_ANSWERS_MODEL_PATH,  "/saiga_mistral_7b_gguf")
        print(f"finished {INCORRECT_ANSWERS_MODEL_PATH}")


@app.function(image=image, volumes={MODEL_DIR: volume_with_models})
@asgi_app()
def entrypoint():
    return web_app