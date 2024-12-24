import wikipedia

# установка языка для Wikipedia - Русский
wikipedia.set_lang("ru")


def get_wikipedia(keywords: list[str]):
    """
    Получает краткое описание из Википедии по ключевым словам, переданным в запросе.
    Если ключевые слова не найдены, возвращается сообщение об ошибке.

    Аргументы:
    keywords -- список ключевых слов для поиска в Википедии

    Возвращает:
    Строку с кратким описанием статьи из Википедии или ошибку в случае неоднозначности.
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
