from django.http import HttpRequest, HttpResponse
from django.shortcuts import render, redirect
from quiz_gen_django.question_generating import (
    get_generator_models,
    gen_all_question_and_answers,
)

generator = get_generator_models()

def home(request: HttpRequest) -> HttpResponse:
    return render(request, "home.html")


def generate_quiz(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        topic = request.POST.get("topic").lower()
        print("[INFO] Topic is:", topic)
        question_answers = gen_all_question_and_answers(
            generator=generator,
            context=topic
        )

        print(question_answers)

        if isinstance(question_answers, str):
            # Если вопрос-ответ вернулся в виде строки (ошибка генерации)
            return render(
                request,
                "error.html",
                {"topic": topic, "question_answers": question_answers},
            )

        elif isinstance(question_answers, list):
            # Если вопрос-ответ вернулся в виде списка (успешно сгенерировано)
            for item in question_answers:
                item["answers"] = item["answers"]
            return render(
                request,
                "quiz.html",
                {"topic": topic, "question_answers": question_answers},
            )
    else:
        # Перенаправление на главную
        return redirect("home")
