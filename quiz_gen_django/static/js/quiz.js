function checkAnswers() {
    const form = document.getElementById("quiz-form");
    const answers = form.querySelectorAll("input[type='radio']:checked");
    let correctCount = 0;
    const totalQuestions = document.querySelectorAll(".question-item").length;

    answers.forEach((answer, index) => {
        const parentDiv = answer.closest(".question-item");
        const correctAnswer = parentDiv.dataset.correctAnswer;
        const selectedAnswer = answer.value;

        // Проверка правильности ответа
        if (selectedAnswer === correctAnswer) {
            correctCount++;
            parentDiv.classList.add("correct");  // Добавляем класс для правильного ответа
            parentDiv.classList.remove("incorrect");  // Убираем класс для неправильного ответа
        } else {
            parentDiv.classList.add("incorrect");  // Добавляем класс для неправильного ответа
            parentDiv.classList.remove("correct");  // Убираем класс для правильного ответа
        }
    });

    // Блокируем все радио-кнопки после проверки
    const allInputs = form.querySelectorAll("input[type='radio']");
    allInputs.forEach(input => {
        input.disabled = true; // Блокировка выбора
    });

    // Отображение результата
    const result = document.getElementById("result");
    result.textContent = `Вы ответили правильно на ${correctCount} из ${totalQuestions} вопросов`;
}




