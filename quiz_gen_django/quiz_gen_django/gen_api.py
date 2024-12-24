from quiz_gen_django.gen_abc import QuestionGeneratorABC


class ApiQuestionGenerator(QuestionGeneratorABC):
    def __init__(self, url: str):
        raise NotImplementedError()
        
    def generate_questions(
        self,
        text: str
    ) -> tuple[str, list[str]]:
        raise NotImplementedError()

    def generate_correct_answers(
        self,
        context: str,
        question: str
    ) -> str:
        raise NotImplementedError()

    def generate_incorrect_answers(
        self,
        context: str,
        question: str,
        correct_answer: str
    ) -> list[str]:
        raise NotImplementedError()
