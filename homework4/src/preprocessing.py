import os
from catboost.datasets import titanic

# Загружаем датасет
titanic_train, titanic_test = titanic()

titanic_train = titanic_train[['Pclass', 'Sex', 'Age']]

# Создание папки datasets, если она не существует
os.makedirs('datasets', exist_ok=True)

# Сохраняем датасет
titanic_train.to_csv('datasets/titanic_train.csv', index=False)