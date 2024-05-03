# Импортируем библиотеки
import os
import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Загрузка датасета
iris = load_iris()
X,y = iris['data'], iris['target']

#Разбиение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Создание папки data, если она не существует
os.makedirs('data', exist_ok=True)

# Записываем датасет в CSV
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df.to_csv('data/iris_dataset.csv', index=False)

#Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Сохранение модели
with open ('/app/data/iris_mode.pkl', 'wb') as f:
    pickle.dump (model,f)