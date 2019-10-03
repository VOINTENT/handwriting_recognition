# Распознавание рукописных цифр

Прогрмма разбита на 3 модуля.

<b>network.py</b>
Модуль для обучения модели нейронной сети и сохранения ее состояния по умолчанию в файле <i>model.h5</i>. Модель нейросети содержит в себе 784 значения на входном слою, три внутренних слоя, с функциями активации 'Relu' на первых двух, 'Softmax' на третьем, и дропаут в качестве функции регуляризации. В завершающем слою содержится 10 нейронов, каждый из которых отождествляет собой один из классов принадлежности.
Запускается из консоли с целью обучения модели нейросети командой <b>python network.py</b> без параметров.

<b>data.py</b>
Модуль для работы с данными, загрузкой их из встроенной библиотеки mnist, имортирование изображений и перевод в числовой вид, формирования confusion matrix, сохранения результатов работы модели в формате json и отображения изображений в числовом и графическом представлении.

<b>main.py</b>
Модуль, содержащий скрипт для работы с предварительно обученной моделью нейросети.
Если скрипт запускается без параметров командной строки, в качестве тестовых данных будут браться стандартные данные из модуля mnist.
<b>python main.py</b>

Если скрипту был передан аргумент в командной строке, то данное значение будет использоваться им в качестве директории, в которой хранятся изображения для тестирования.
<b>python main.py data</b>

В обоих случаях результаты будут сохранены в файле <i>data.json</i>, а на экран будет выведена confusion matrix.