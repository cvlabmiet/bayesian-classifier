#coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from data import getTrainingSetYCrCb, getTestSetYCrCb


"""
Вычисление ячейки гистограммы, к которой относится заданная точка.

Args:
    bins: массив левых границ ячеек разбиения по каждой из компонент.
    pixel: точка изображения.

Returns:
    Позиция ячейки, соответствующей точки изображения.
"""
def getPosition(bins, pixel):
    ndims = len(bins)
    position = np.empty(ndims, dtype = int)

    for dim in np.arange(ndims):
        current_bins = bins[dim]
        last_bin = current_bins[-1]

        if pixel[dim] >= last_bin:
            position[dim] = len(current_bins) - 1
        else:
            res = current_bins[:] > pixel[dim]
            position[dim] = np.argmax(res) - 1

    position = tuple(position)
    return position


"""
Построение гистограммы данных с заданными параметрами ячеек.

Args:
    bins: массив границ ячеек разбиения по каждой из компонент.
    data: данные, по которым строится гистограмма.

Returns:
    Массив, соответствующий гистограмме.
"""
def getHistogram(bins, data):
    ndims = len(bins)
    data_length = data.shape[0]
    dims = np.empty(ndims, dtype = int)

    for i in np.arange(ndims):
        dims[i] = bins[i].size

    hist = np.zeros(dims)

    for i in np.arange(data_length):
        current = data[i,:]
        position = getPosition(bins, current)
        if min(position) < 0:
            return -1

        hist[position] += 1

    if hist.sum() != 0:
        hist = hist / hist.sum()

    return hist


"""
Алгоритм выделения кожи на изображении на основе классификатора Байеса.

Attributes:
    model_path: каталог, содержащий натренированные модели классификатора.
    bins: массив границ ячеек разбиения по каждой из компонент.
    skin_histogram: гистограмма, соответствующая точкам кожи.
    non_skin_histogram: гистограмма, соответствующая точкам фона.
"""
class Bayes(object):
    model_path = "bayes_models\\"
    bins = None
    skin_histogram = None
    non_skin_histogram = None

    """Инициализация модели алгоритма."""
    def __init__(self, bins, skin, non_skin):
        self.bins = bins
        ndims = len(bins)
        steps = np.empty(ndims, dtype = int)

        for i in np.arange(ndims):
            steps[i] = bins[i][1] - bins[i][0]

        postfix = ""
        for i in np.arange(ndims):
            postfix += "_" + str(steps[i])

        skin_model_name = self.model_path + "bayes_skin_model" + postfix + ".npy"
        non_skin_model_name = self.model_path + "bayes_non_skin_model" + postfix + ".npy"

        try:
            # Загрузка сохранённой модели.
            self.skin_histogram = np.load(skin_model_name)
            self.non_skin_histogram = np.load(non_skin_model_name)
        except Exception:
            # Обучение новой модели.
            self.skin_histogram = getHistogram(bins, skin)
            self.non_skin_histogram = getHistogram(bins, non_skin)
            np.save(skin_model_name, self.skin_histogram)
            np.save(non_skin_model_name, self.non_skin_histogram)

    """ Построение карты кожи изображения."""
    def apply(self, image):
        rows, cols, _ = image.shape
        skin_map = np.empty([rows, cols])

        for y in range(rows):
            for x in range(cols):
                current = image[y,x,:]
                position = getPosition(self.bins, current)
                skin = self.skin_histogram[position]
                non_skin = self.non_skin_histogram[position]
                if non_skin == 0:
                    skin_map[y,x] = skin
                else:
                    skin_map[y,x] = skin / non_skin

        return skin_map

    """Построение бинарной маски сегментации кожи."""
    def classify(self, image, threshold):
        skin_map = self.apply(image)
        skin_map[skin_map <= threshold] = 0
        skin_map[skin_map > threshold] = 255
        skin_map = np.uint8(skin_map)
        return skin_map
