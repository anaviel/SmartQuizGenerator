import spacy
import wikipedia
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Загрузка модели русского языка для обработки текста (SpaCy)
nlp = spacy.load("ru_core_news_sm")

# Устанавливаем язык для Wikipedia (русский)
wikipedia.set_lang("ru")

# Загружаем модели и токенизаторы:
# Модель для генерации вопросов
question_model = T5ForConditionalGeneration.from_pretrained('/home/morph/proj/SmartQuizGenerator/quiz_gen_django/quiz_gen_django/question_generation_model')
# Модель для генерации ответов
answer_model  = T5ForConditionalGeneration.from_pretrained('/home/morph/proj/SmartQuizGenerator/quiz_gen_django/quiz_gen_django/final_model')
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
        return f"Запрос слишком неоднозначен. Возможные страницы: {e.options}"  # В случае неоднозначности


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

    # Печатаем результат из Википедии для проверки
    print("Текст из Википедии:", wiki_summary)

    # Если Wikipedia не дала информации, возвращаем пустой список вопросов
    if wiki_summary == "Информация по запросу не найдена.":
        return ["Не удалось найти подходящую информацию."]

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
        num_beams=5,
        num_return_sequences=5,  # Генерация 5 различных вопросов
        temperature=1.2,  # Используем более высокую температуру для разнообразия
        top_p=0.9,  # Используем отбор по вероятности
        early_stopping=True,  # Останавливаем генерацию, когда модель "уверена" в результате
    )

    questions = [
        tokenizer.decode(q, skip_special_tokens=True) for q in question_ids
    ]  # Декодируем вопросы в текст

    return questions


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
    questions = generate_questions(text)

    # Генерируем ответы на вопросы
    answers = generate_answers(text, questions)

    # Возвращаем вопросы и ответы как список пар (вопрос, ответ)
    question_answer = list(zip(questions, answers))

    return question_answer
    # return question_answer[:5]
