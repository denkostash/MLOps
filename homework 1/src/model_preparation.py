# Импортируем библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Подготовка и обучение модели
train_data = []
for i in range(1):
    try:
        data = pd.read_csv(f'train/train_preprocessed/preprocessed_data{i}.csv')
        train_data.append(data)
    except FileNotFoundError:
        print(f"Файл train/train_preprocessed/preprocessed_data{i}.csv не найден.")

X = pd.concat(train_data, ignore_index=True).drop('temperature', axis=1)
y = pd.concat(train_data, ignore_index=True)['temperature']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
