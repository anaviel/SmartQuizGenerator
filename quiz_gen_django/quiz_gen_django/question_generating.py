import os
from quiz_gen_django.settings   import INFERENCE_SERVER_MODE
from quiz_gen_django.gen_abc    import QuestionGeneratorABC
from quiz_gen_django.gen_local  import LocalQuestionGenerator
from quiz_gen_django.gen_api    import ApiQuestionGenerator


def get_generator_models() -> QuestionGeneratorABC:
    print("[DEBUG] Mode is:", INFERENCE_SERVER_MODE)
    if INFERENCE_SERVER_MODE == "API":
        api_url_base = os.getenv("API_URL_BASE")
        print("[DEBUG] API_URL_BASE is:", api_url_base)
        return ApiQuestionGenerator(api_url_base)

    qst_model_path                  = os.getenv("MODEL_QUESTIONS_PATH")
    ans_model_path                  = os.getenv("MODEL_CORRECT_ANSWERS_PATH")
    ans_incorrect_model_path_gguf   = os.getenv("MODEL_INCORRECT_ANSWERS_PATH")
    
    return LocalQuestionGenerator(
        questions_model_path=qst_model_path,
        answers_model_path=ans_model_path,
        incorrect_answers_model_path=ans_incorrect_model_path_gguf
    )


def gen_all_question_and_answers(
    generator: QuestionGeneratorABC,
    context: str
) -> list[dict] | str:
    """
    Основная функция, которая генерирует вопросы и ответы для переданного контекста.

    Аргументы:
    context -- контекст, на основе которого генерируются вопросы и ответы

    Возвращает:
    Список пар (вопрос, ответ) в виде двумерного массива.
    """
    # генерация вопросов
    wiki_summary, questions = generator.generate_questions(context)
    # TODO: проверить что этот кейс обрабатывается правильно
    if questions is None:
        return wiki_summary

    # генерация ответов
    qst_and_correct_incorrect_ans = generator.generate_all_answers(
        wiki_summary,
        questions
    )
    print(f"[INFO] Сгенерированные вопросы и ответы: {qst_and_correct_incorrect_ans}")
    return qst_and_correct_incorrect_ans
