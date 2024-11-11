import spacy
import wikipedia

# Загрузка модели русского языка
nlp = spacy.load('ru_core_news_sm')

# Устанавливаем язык для Wikipedia (русский)
wikipedia.set_lang('ru')

# Функция для получения краткой информации по ключевому слову из Wikipedia
def get_wikipedia(keywords):
    # Объединяем ключевые слова в строку для поиска
    search_query = ' '.join(keywords)
    print(f"Ищем информацию по запросу: {search_query}")

    # Ищем статьи по запросу
    try:
        search_results = wikipedia.search(search_query, results=1)  # Получаем 1 результат
        print(f"Результаты поиска: {search_results}")

        if search_results:
            # Получаем описание страницы (по первому результату)
            page_title = search_results[0]
            page = wikipedia.page(page_title)  # Получаем страницу по заголовку
            return page.summary  # Возвращаем краткое описание страницы
        else:
            return "Информация по запросу не найдена."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Запрос слишком неоднозначен. Возможные страницы: {e.options}"  # В случае неоднозначности

# Функция генерации вопросов и ответов
def question_gen(text):
    # Обрабатываем текст с помощью модели SpaCy
    nlp_text = nlp(text)

    # Извлекаем ключевые слова (существительные, прилагательные, глаголы)
    keywords = [token.text for token in nlp_text if token.pos_ in ['NOUN', 'ADJ', 'VERB']]

    # Получаем информацию по ключевым словам
    wiki_summary = get_wikipedia(keywords) if keywords else "Нет ключевых слов для поиска."

    # Передаем информацию в функцию генерации вопроса и ответа
    question_answer = model_generate(wiki_summary)

    return question_answer


def model_generate(wiki_summary):
    # Пустая функция, которая пока что только возвращает переменные
    # Это место будет заменено на нейронную модель, которая будет формировать вопрос и ответ.

    # Временные переменные для проверки сбора данных:
    question_answer = [["Какие задачи могут выполнять интеллектуальные машины, созданные в рамках AI?", "Распознавание образов"],
                       ["Что представляет собой машинное обучение в AI?", "Алгоритмы, позволяющие машинам учиться на основе данных и улучшать свою производительность со временем без явного программирования"],
                       ["Зачем используется обработка естественного языка в AI?", "Для понимания и интерпретации человеческого языка"],
                       ["Что делает компьютерное зрение в рамках AI?", "Облегчает распознавание изображений и обнаружение объектов"]]

    return question_answer[:5]

