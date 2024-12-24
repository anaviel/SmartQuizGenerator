from typing import Sequence
from quiz_gen_django.search_wikipedia import get_wikipedia

import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def generate_correct_answers(
    tokenizer: PreTrainedTokenizerBase,
    answer_model: PreTrainedModel,
    context: str,
    question: str
) -> str:
    """
    Генерирует ответы на переданные вопросы с использованием контекста.

    Аргументы:
    context -- текст, который служит контекстом для ответов на вопросы
    questions -- список вопросов, на которые нужно сгенерировать ответы

    Возвращает:
    Список ответов на каждый вопрос.
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
        get_wikipedia(keywords) if keywords else "Нет ключевых слов для поиска."
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
