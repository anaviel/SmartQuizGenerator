from django.shortcuts import render
from quiz_gen_django.question_generating import question_gen


def home(request):
    return render(request, 'home.html')

def qenerate_quiz(request):
    if request.method == 'POST':
        topic = request.POST.get('topic')
        question_answers = question_gen(topic)
        print(question_answers)
        if isinstance(question_answers, str):
            return render(request, 'error.html', {'topic': topic, 'question_answers': question_answers})
        elif isinstance(question_answers, list):
            for item in question_answers:
                item['answers'] = item['answers']
            return render(request, 'quiz.html', {'topic': topic, 'question_answers': question_answers})

