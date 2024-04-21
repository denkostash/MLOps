# Импортируем библиотеки
import os
import pandas as pd
from model_preparation import model
from sklearn.metrics import mean_squared_error

# Тестирование модели
test_data = []
for i in range(len(os.listdir('test/test_preprocessed'))):
    data = pd.read_csv(f'test/test_preprocessed/preprocessed_data{i}.csv')
    test_data.append(data)

X = pd.concat(test_data, ignore_index=True).drop('temperature', axis=1)
y = pd.concat(test_data, ignore_index=True)['temperature']

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
