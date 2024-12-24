from abc import ABC, abstractmethod
import random


class QuestionGeneratorABC(ABC):
    @abstractmethod
    def generate_questions(
        self,
        text: str
    ) -> tuple[str, list[str]]:
        pass

    @abstractmethod
    def generate_correct_answers(
        self,
        context: str,
        question: str
    ) -> str:
        pass

    @abstractmethod
    def generate_incorrect_answers(
        self,
        context: str,
        question: str,
        correct_answer: str
    ) -> list[str]:
        pass

    def generate_all_answers(
        self,
        context: str,
        questions: list[str]
    ) -> list[dict]:
        
        answers = []
        
        for question in questions:
            correct_answer = self.generate_correct_answers(
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
