# MLOps. Практическое задание №4

Выполните все основные операции с dvc и закрепите полученные теоретические знания практическими действиями.

## Структура файлов и каталогов

- preprocessing.py
   - Загрузка датасета Titanic
   - Предобработка тренировочного набора данных
   - Создание папки data, если она не существует
   - Запись датасета в CSV

- preproccessing_null_values.py
   - Загрузка датасета Titanic
   - Вычисляем средний возраст
   - Заменяем пропущенные значения средним возрастом
   - Сохраняем обновленный фреймдейт данных
   
- preprocessing_one_hoc.py   
   - Загрузка датасета Titanic
   - Создаем кодировщик One-Hot Encoder
   - Кодируем признак пола
   - Преобразуем закодированный массив в фреймдейт данных
   - Объединяем закодированные и исходные данные
   - Сохраняем обновленный фреймдейт данных
     
- datasets.dvc
   - Информация о файле или каталоге в системе управления данными DVC

- dvc_commands.txt
   - Используется для добавления удаленного репозитория в проект DVC

