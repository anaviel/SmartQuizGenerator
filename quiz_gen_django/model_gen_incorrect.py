import re
from llama_cpp import Llama

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