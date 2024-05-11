import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from palmerpenguins import load_penguins
from sklearn.ensemble import RandomForestClassifier

# Загружаем данные
penguins = load_penguins()

# Создание папки data, если она не существует
os.makedirs('data', exist_ok=True)

# Сохраняем наборы данных в файл CSV
penguins.to_csv("data/penguins.csv", index=False)

# Преобразуем категориальные столбцы в числовые
penguins['island'] = pd.Categorical(penguins['island'])
penguins['sex'] = pd.Categorical(penguins['sex'])
penguins = pd.get_dummies(penguins, columns=['island', 'sex'])

# Обработка пропущенных значений
penguins.dropna(inplace=True)

# Разделяем данные на обучающие и тестовые наборы
X = penguins.drop('species', axis=1)
y = penguins['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Оцениваем модель на тестовом множестве
score = model.score(X_test, y_test)
print('Оценка модели:', score)

# Создаем новые данные о характеристиках пингвинов
new_penguins = pd.DataFrame({
    'island': ['Torgersen'],
    'bill_length_mm': [34.1],
    'bill_depth_mm': [18.1],
    'flipper_length_mm': [193],
    'body_mass_g': [3475],
    'sex': ['male'],
    'year': ['2007']
})

###примеры для заполнения выше
#1. тип - Gentoo,      данные: Biscoe      46.1, 15.1, 215.0, 5100.0, male, 2007
#2. тип - Chinstrap,   данные: Dream       49.5, 19.0, 200.0, 3800.0, male, 2008
#3. тип - Adelie,      данные: Torgersen   34.1, 18.1, 193.0, 3475.0, male, 2007

# Преобразуем категориальные столбцы в числовые
new_penguins['island'] = pd.Categorical(new_penguins['island'])
new_penguins['sex'] = pd.Categorical(new_penguins['sex'])
new_penguins = pd.get_dummies(new_penguins, columns=['island', 'sex'])

# Создаем фиктивные столбцы для столбцов, отсутствующих в обученной модели
new_penguins['island_Dream'] = np.zeros(new_penguins.shape[0])
new_penguins['island_Biscoe'] = np.zeros(new_penguins.shape[0])
new_penguins['island_Torgersen'] = np.zeros(new_penguins.shape[0])
new_penguins['sex_male'] = np.zeros(new_penguins.shape[0])
new_penguins['sex_female'] = np.zeros(new_penguins.shape[0])

# Обработка пропущенных значений
new_penguins.dropna(inplace=True)

# **Сортируем данные по столбцам так же, как в обучающих данных**
new_penguins = new_penguins[X_train.columns]

# Делаем предсказания для новых данных
y_pred = model.predict(new_penguins)

# Выводим предсказания
print('Предполагаемый вид пингвина: ', y_pred)

