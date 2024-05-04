# MLOps. Практическое задание №5

Применить средства автоматизации тестирования python для автоматического тестирования качества работы модели машинного обучения на различных датасетах.

## Структура файлов и каталогов

- dataframe.py
   - Создаем набор 3 набора данных
   - Создаем папку для хранения этих наборов
   - Сохраняем наборы данных в файл CSV

- Training.py
   - Загрузка датасета
   - Разделяем данные на признаки и целевую переменную
   - Создаем и обучаем модель линейной регрессии
   - Добавляем шум к целевой переменной
   - Сохраняем набор данных в файл CSV
   
- test.py   
   - Загружаем 3 датасета с качественными данными и  1 шумовый датасет
   - Создаем функцию для тестрирования, в которой вычисляем и выводим MSE
