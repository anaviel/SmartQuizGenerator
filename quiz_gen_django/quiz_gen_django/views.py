from django.shortcuts import render
from quiz_gen_django.question_generating import question_gen


def home(request):
    return render(request, 'home.html')

def qenerate_quiz(request):
    if request.method == 'POST':
        topic = request.POST.get('topic')
        # -- Логика генерации вопросов --
        question_answer = question_gen(topic)

        if isinstance(question_answer, str):
            return render(request, 'error.html', {'topic': topic, 'question_answer': question_answer})

        elif isinstance(question_answer, list):
            return render(request, 'quiz.html', {'topic': topic, 'question_answer': question_answer})
