from django.shortcuts import render
from quiz_gen_django.question_generating import question_gen


def home(request):
    return render(request, 'home.html')

def qenerate_quiz(request):
    if request.method == 'POST':
        topic = request.POST.get('topic')
        # -- Логика генерации вопросов --
        question = question_gen(topic)
        new_question = ''
        for i in question:
             new_question = new_question  + i + ' '
        new_question += '?'

        return render(request, 'quiz.html', {'topic': topic, 'question': new_question})
