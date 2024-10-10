import spacy
import wikipediaapi

# Загрузка модели русского языка
nlp = spacy.load('ru_core_news_sm')


# Функция для получения краткой информации по ключевому слову из Wikipedia
def get_wikipedia(keyword):
    # Указываем 'user_agent' при создании экземпляра Wikipedia
    wiki_wiki = wikipediaapi.Wikipedia(
        language='ru',
        user_agent='SmartQuizGenerator/1.0 (example@example.com)'  # Ваш User-Agent
    )

# Функция генерации вопросов и ответов
def question_gen(text):
    nlp_text = nlp(text)
    keywords = [token.text for token in nlp_text if token.pos_ in ['NOUN', 'ADJ', 'VERB']]
    # token - один из элементов, полученных после разбиения текста
    # .text - строковое представление токена
    # .pos - часть речи (NOUN - прилагательное, ADJ - существительное, VERB - глагол)
    get_wikipedia(keywords)

    return keywords

