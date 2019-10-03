"""
Модуль для работы с данными, загрузкой их из встроенной библиотеки mnist,
имортирование изображений и перевод в числовой вид, формирования confusion matrix,
сохранения результатов работы модели в формате json и отображения изображений в числовом
и графическом представлении
"""

from tensorflow import keras

import numpy as np
import PIL, glob, json
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_data(test_percent=0.13):
    """
    Загрузка датасета с разделением на тренировочную и тестовую часть по
    указанному проценту и преобразованием данных в диапазон [0..1]
    """
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    all_images = np.append(train_images, test_images, axis=0)
    all_labels = np.append(train_labels, test_labels, axis=0)

    num_test = int( len(all_images) * test_percent )

    test_images = all_images[ : num_test]
    test_labels = all_labels[ : num_test]

    train_images = all_images[num_test : ]
    train_labels = all_labels[num_test : ]

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def get_own_data():
    """
    Предполагается, что на момент вызова метода, интерпретатор находится в папке с
    необходимыми изображениями. Считываются изображения, обрабатываются, и выдаются
    в виде числовых массивов. Истинным значением каждого изображения считается его
    название, не считая расширения
    """
    test_images = []
    test_labels = []
    for file in glob.glob("*.png"):
        img = PIL.Image.open(file).convert('L')
        imgarr = np.array(img)
        for i in range( len(imgarr) ):
            for j in range( len(imgarr[i]) ):
                imgarr[i][j] = 255 - imgarr[i][j]
        test_images.append(imgarr)
        test_labels.append( int(file.split('.')[0]) )

    test_labels = np.array(test_labels)
    test_images = np.array(test_images)

    return test_images, test_labels

def get_conf_mat(test_predict, test_labels):
    """
    Формирование confusion matrix
    """
    test_predict = list(map(np.argmax, test_predict) )
    con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=test_predict).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    con_mat_df = pd.DataFrame(con_mat_norm,
                         index = classes,
                         columns = classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_json(test_predict, test_labels):
    """
    Записывает результат работы модели в формате json в файл
    по умолчанию 'data.json', находящийся в одной папке с
    вызванным скриптом
    """
    FILENAME = 'data.json'
    data = []
    for i in range( len(test_predict) ):
        res = {
            'Ожидалось:' : str(np.argmax(test_predict[i])),
            'Реальное значение' : str(test_labels[i]),
            'Статистика' : list(map(str, test_predict[i]))
        }
        data.append(res)

    with open('data.json', 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def show_dimage(image):
    """
    Пример цифрового отображения цифры
    """
    if image[0][0] is np.float64:
        image = image * 255

    for i in range( len( image ) ):
        for j in range( len( image[i] ) ):
            max_len = len( str( max( image[:,j] ) ) )
            spaces = ( max_len - len( str( image[i][j] ) ) + 1 ) * ' '
            print(int(image[i][j]), end = spaces)
        print()

def show_gimage(image):
    """
    Пример графического отображения цифры
    """
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()
