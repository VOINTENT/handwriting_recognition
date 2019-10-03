"""
Модуль для обучения модели нейронной сети и сохранения ее состояния
"""

import data

from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Файл сохранения модели сети
FILENAME = 'model.h5'

if __name__ == '__main__':
    # Загрузка данных
    (train_images, train_labels), (_, _) = data.get_data(test_percent=0.13)

    # Настройка слоев модели
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Компиляция модели
    model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

    # Просмотр структуры получившейся сети
    model.summary()

    # Обучение сети
    model.fit(train_images, train_labels, epochs=10)

    while True:
        answer = str.lower( input('Сохранить модель? y/n: ') )
        if answer == 'y':
            # Сохранение модели сети
            model.save(FILENAME)
            break
        elif answer == 'n':
            break
