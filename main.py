"""
Если скрипт запускается без параметров командной строки, в качестве тестовых данных
будут браться стандартные данные из модуля mnist. Если скрипту был передан аргумент в
командной строке, то данное значение будет использоваться в качестве директории,
в которой хранятся изображения для тестирования.
"""

import network as nw
import data

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, glob, os


def evaluate(model, test_images, test_labels):
    """
    Запуск обученной сети на тестовых данных
    """
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nТочность на проверочных данных', test_acc)

if __name__ == '__main__':
    # Загрузка модели
    model = keras.models.load_model(nw.FILENAME)

    if len(sys.argv) > 1:
        # Загрузка собственноручно созданных изображений
        dir = sys.argv[1]
        os.chdir(os.getcwd() + '/' + dir)
        test_images, test_labels = data.get_own_data()
        os.chdir('..')

        test_predict = model.predict(test_images)
    else:
        # Загрузка стандартных тестовых данных
        (_, _), (test_images, test_labels) = data.get_data()

    evaluate(model, test_images, test_labels)

    test_predict = model.predict(test_images)

    data.get_conf_mat(test_predict, test_labels)
    data.get_json(test_predict, test_labels)
