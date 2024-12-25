import random
import httpx
from quiz_gen_django.gen_abc import QuestionGeneratorABC


class ApiQuestionGenerator(QuestionGeneratorABC):
    def __init__(self, url: str):
        """
        Инициализирует генератор вопросов, взаимодействующий с API.

        Args:
            url (str): Базовый URL API.
        """
        self.base_url = url
        self.client = httpx.Client(base_url=self.base_url)

    def generate_questions(self, text: str) -> tuple[str, list[str]]:
        """
        Генерирует вопросы, запрашивая их из API.

        Args:
            text (str): Текст для генерации вопросов.

        Returns:
            tuple[str, list[str]]: Краткий контекст и список вопросов.
        """
        response = self.client.get(
            "/generate/generate_questions", params={"topic": text}, timeout=130
        )
        response.raise_for_status()
        data = response.json()
        return data["wiki_summary"], data["questions"]

    def generate_correct_answer(self, context: str, question: str) -> str:
        """
        Генерирует правильный ответ, запрашивая его из API.

        Args:
            context (str): Контекст вопроса.
            question (str): Вопрос.

        Returns:
            str: Правильный ответ.
        """
        response = self.client.get(
            "/generate/generate_correct_answer",
            params={"context": context, "question": question},
            timeout=130,
        )
        response.raise_for_status()
        data = response.json()
        return data["answer_correct"]

    def generate_incorrect_answers(
        self, context: str, question: str, correct_answer: str
    ) -> list[str]:
        """
        Генерирует неправильные ответы, запрашивая их из API.

        Args:
            context (str): Контекст вопроса.
            question (str): Вопрос.
            correct_answer (str): Правильный ответ.

        Returns:
            list[str]: Список неправильных ответов.
        """
        response = self.client.get(
            "/generate/generate_incorrect_answers",
            params={
                "context": context,
                "question": question,
                "correct_answer": correct_answer,
            },
            timeout=130,
        )
        response.raise_for_status()
        data = response.json()
        return data["incorrect_answers"]

    def generate_all_answers(
        self,
        context: str,
        questions: list[str],
    ) -> list[dict]:
        
        answers = []
        
        for question in questions:
            correct_answer = self.generate_correct_answer(
                context,
                question
            )

            incorrect_answers = self.generate_incorrect_answers(
                context,
                question,
                correct_answer
            )

            all_answers = [correct_answer] + incorrect_answers
            random.shuffle(all_answers)
            answers.append(
                {
                    "question": question,
                    "answers": all_answers,
                    "correct_answer": correct_answer,
                }
            )

        return answers
