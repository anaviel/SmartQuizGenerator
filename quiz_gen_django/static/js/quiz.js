document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('topic-form');
    const loadingScreen = document.getElementById('loading-screen');

    // При отправке формы показываем экран загрузки
    if (form) {
        form.addEventListener('submit', (e) => {
            // Показываем экран загрузки
            if (loadingScreen) {
                loadingScreen.style.display = 'flex';
            }
        });
    }
});


function checkAnswers() {
    const form = document.getElementById("quiz-form");
    const answers = form.querySelectorAll("input[type='radio']:checked");
    let correctCount = 0;
    const totalQuestions = document.querySelectorAll(".question-item").length;

    answers.forEach(answer => {
        const parentDiv = answer.closest(".question-item");
        const correctAnswer = parentDiv.dataset.correctAnswer;
        const selectedAnswer = answer.value;

        // Проверка правильности ответа
        if (selectedAnswer === correctAnswer) {
            correctCount++;
            parentDiv.classList.add("correct");
            parentDiv.classList.remove("incorrect");
        } else {
            parentDiv.classList.add("incorrect");
            parentDiv.classList.remove("correct");

            // Ищем правильный ответ и выделяем его фоном
            const correctAnswerElem = parentDiv.querySelectorAll("input[type='radio']");
            correctAnswerElem.forEach(input => {
               if (input.value === correctAnswer) {
                    const label = input.nextElementSibling;
                    label.style.backgroundColor = '#c8e6c9';
                    label.style.fontWeight = '350';
                    label.style.color = 'black';
                }
            });
        }
    });

    // Отключаем все радиокнопки после проверки
    const allInputs = form.querySelectorAll("input[type='radio']");
    allInputs.forEach(input => {
        input.disabled = true;
    });

    const result = document.getElementById("result");
    result.textContent = `Вы ответили правильно на ${correctCount} из ${totalQuestions} вопросов`;
}

