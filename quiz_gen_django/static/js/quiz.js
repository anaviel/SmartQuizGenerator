function toggleAnswer(button) {
    var answer = button.nextElementSibling; // Соседний элемент с классом "answer"
    
    // Добавляем или убираем класс "active" у кнопки
    button.classList.toggle("active");
    
    // Показываем или скрываем ответ
    if (button.classList.contains("active")) {
        answer.style.display = "block";
        button.textContent = "Скрыть ответ";
    } else {
        answer.style.display = "none";
        button.textContent = "Показать ответ";
    }
}

