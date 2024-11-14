import spacy
import wikipedia
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

# Загрузка модели русского языка для обработки текста (SpaCy)
nlp = spacy.load("ru_core_news_sm")

# Инициализация модели для эмбеддингов
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Устанавливаем язык для Wikipedia (русский)
wikipedia.set_lang("ru")

# Загружаем модели и токенизаторы:
script_dir = os.path.dirname(os.path.abspath(__file__))

relative_path_1 = os.path.join(script_dir, "question_generation_model")
relative_path_2 = os.path.join(script_dir, "final_model")


# Модель для генерации вопросов
question_model = T5ForConditionalGeneration.from_pretrained(relative_path_1)

# Модель для генерации ответов
answer_model  = T5ForConditionalGeneration.from_pretrained(relative_path_2)

# Токенизатор для обеих моделей
tokenizer = T5Tokenizer.from_pretrained("ai-forever/ruT5-base")


# Функция для получения краткой информации по ключевым словам из Wikipedia
def get_wikipedia(keywords):
    """
    Получает краткое описание из Википедии по ключевым словам, переданным в запросе.
    Если ключевые слова не найдены, возвращается сообщение об ошибке.

    Аргументы:
    keywords -- список ключевых слов для поиска в Википедии

    Возвращает:
    Строку с кратким описанием статьи из Википедии или ошибку в случае неоднозначности.
    """
    search_query = " ".join(keywords)  # Объединяем ключевые слова в строку для поиска
    print(f"Ищем информацию по запросу: {search_query}")

    # Ищем статьи по запросу
    try:
        search_results = wikipedia.search(
            search_query, results=1
        )  # Ищем 1 статью по запросу
        print(f"Результаты поиска: {search_results}")

        if search_results:
            # Получаем описание страницы по первому результату поиска
            page_title = search_results[0]
            page = wikipedia.page(
                page_title
            )  # Загружаем страницу по найденному заголовку
            return page.summary  # Возвращаем краткое описание статьи
        else:
            return "Информация по запросу не найдена."
    except wikipedia.exceptions.DisambiguationError as e:
        return "Запрос слишком неоднозначен."  # В случае неоднозначности


# Функция для генерации вопросов из текста
def generate_questions(text):
    """
    Генерирует вопросы на основе переданного текста.

    Аргументы:
    text -- текст, на основе которого модель генерирует вопросы

    Возвращает:
    Список сгенерированных вопросов.
    """
    nlp_text = nlp(text)  # Обрабатываем текст с помощью модели SpaCy

    # Извлекаем ключевые слова (существительные, прилагательные, глаголы) из текста
    keywords = [
        token.text for token in nlp_text if token.pos_ in ["NOUN", "ADJ", "VERB"]
    ]

    # Получаем краткую информацию по ключевым словам из Википедии
    wiki_summary = (
        get_wikipedia(keywords) if keywords else "Нет ключевых слов для поиска."
    )

    if wiki_summary == "Информация по запросу не найдена.":
        question_text = ''
        return wiki_summary, question_text

    if wiki_summary == "Запрос слишком неоднозначен.":
        question_text = ''
        return wiki_summary, question_text

    # Печатаем результат из Википедии для проверки
    print("Текст из Википедии:", wiki_summary)

    # Генерация вопросов с использованием обученной модели для вопросов
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
        num_return_sequences=12,  # Генерация 12 различных вопросов
        temperature=1.2,  # Используем более высокую температуру для разнообразия
        top_p=0.9,  # Используем отбор по вероятности
        early_stopping=True,  # Останавливаем генерацию, когда модель "уверена" в результате
    )

    # Декодируем вопросы в текст
    questions = [
        tokenizer.decode(q, skip_special_tokens=True) for q in question_ids
    ]

    # Преобразуем вопросы в векторы
    question_embeddings = embedder.encode(questions, convert_to_tensor=True)

    unique_questions = []
    for i, question in enumerate(questions):
        # Сравниваем с ранее добавленными вопросами
        is_unique = True
        for unique_question in unique_questions:
            # Вычисляем схожесть между вопросами
            sim_score = util.pytorch_cos_sim(question_embeddings[i], unique_question[1])
            if sim_score >= 0.7:  # Порог схожести
                is_unique = False
                break

        if is_unique and len(unique_questions) < 5:
            unique_questions.append((question, question_embeddings[i]))

    # Оставляем только вопросы (без векторов)
    return wiki_summary, [q[0] for q in unique_questions]


# Функция для генерации ответов по контексту и вопросам
def generate_answers(context, questions):
    """
    Генерирует ответы на переданные вопросы с использованием контекста.

    Аргументы:
    context -- текст, который служит контекстом для ответов на вопросы
    questions -- список вопросов, на которые нужно сгенерировать ответы

    Возвращает:
    Список ответов на каждый вопрос.
    """
    answers = []
    for question in questions:
        input_text = (
            question + " " + context
        )  # Формируем запрос для модели: вопрос + контекст
        input_ids = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).input_ids
        answer_ids = answer_model.generate(
            input_ids, max_length=128, num_beams=5, early_stopping=True
        )
        answer = tokenizer.decode(
            answer_ids[0], skip_special_tokens=True
        )  # Декодируем ответ в текст
        answers.append(answer)  # Добавляем ответ в список

    return answers


# Функция для генерации вопросов и ответов по контексту
def question_gen(text):
    """
    Основная функция, которая генерирует вопросы и ответы для переданного контекста.

    Аргументы:
    text -- контекст, на основе которого генерируются вопросы и ответы

    Возвращает:
    Список пар (вопрос, ответ) в виде двумерного массива.
    """
    # Генерируем вопросы по контексту
    wiki_summary, questions = generate_questions(text)

    if questions == '':
        return wiki_summary

    # Генерируем ответы на вопросы
    answers = generate_answers(wiki_summary, questions)

    # Возвращаем вопросы и ответы как список пар (вопрос, ответ)
    question_answer = list(zip(questions, answers))

    return question_answer
