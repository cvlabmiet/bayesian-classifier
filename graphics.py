#coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data import getTrainingSetYCrCb, getTestSetYCrCb
from bayes import Bayes


"""
Построение ROC кривой с фиксированным шагом.

Args:
    skin_detector: алгоритм распознавания кожи.
    step: шаг, с которым изменяется параметр алгоритма.

Returns:
    Количество TPR, FPR.
"""
def getROCCurve(skin_detector, step):
    test_skin, test_non_skin = getTestSetYCrCb()

    skin_map = skin_detector.apply(test_skin)
    non_skin_map = skin_detector.apply(test_non_skin)

    TPR = np.array([])
    FPR = np.array([])

    for threshold in np.arange(0, max(skin_map.max(), non_skin_map.max()) + step, step):
        TPR = np.append(TPR, np.sum(skin_map > threshold))
        FPR = np.append(FPR, np.sum(non_skin_map > threshold))

    TPR /= test_skin.shape[0]
    FPR /= test_non_skin.shape[0]

    return TPR, FPR


"""
Функция построения ROC-кривой с максимально допустимым
шагом по оси FPR.

Args:
    skin_detector: алгоритм распознавания кожи.
    step: максимальный шаг между значениями на оси FPR.

Returns:
    Количество TPR, FPR.
"""
def getOptROCCurve(skin_detector, max_step):
    test_skin, test_non_skin = getTestSetYCrCb()

    test_skin_len = np.double(test_skin.shape[0])
    test_non_skin_len = np.double(test_non_skin.shape[0])

    skin_map = skin_detector.apply(test_skin)
    non_skin_map = skin_detector.apply(test_non_skin)

    TPR = []
    FPR = []
    threshold_list = [0, max(skin_map.max(), non_skin_map.max())]
    for threshold in threshold_list:
        TPR += [np.sum(skin_map > threshold) / test_skin_len]
        FPR += [np.sum(non_skin_map > threshold) / test_non_skin_len]

    index = 0
    while index != (len(FPR) - 1):
        step = abs(FPR[index + 1] - FPR[index])

        if step > max_step:
            if abs(threshold_list[index + 1] - threshold_list[index]) < 10**(-10):
                index += 1
                continue

            threshold = (threshold_list[index + 1] + threshold_list[index]) / 2.0
            TPR.insert(index + 1, np.sum(skin_map > threshold) / test_skin_len)
            FPR.insert(index + 1, np.sum(non_skin_map > threshold) / test_non_skin_len)
            threshold_list.insert(index + 1, threshold)
            continue

        index += 1

    TPR = np.array(TPR)
    FPR = np.array(FPR)

    return TPR, FPR


"""
Построение ROC-кривой.

Args:
    TPR: массив значений TPR.
    FPR: массив значений FPR.
    linestyle: тип линии ROC кривой.
    linewidth: толщина линии ROC кривой
"""
def plotROCCurve(TPR, FPR, linestyle='.', linewidth = 10.0):
    plt.plot(FPR, TPR, linestyle=linestyle, linewidth=linewidth, color="Black")
    plt.grid("on")
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set(xlabel = u'FPR', ylabel = u'TPR')
    return


"""
Построение ROC кривых алгоритмов Байеса с различными вариантами обучения.
"""
def plotBayesROCCurves():
    TP_unique_non_skin = np.load("unique_non_skin\\ROC_Curves\\bayes_TP.npy")
    FP_unique_non_skin = np.load("unique_non_skin\\ROC_Curves\\bayes_FP.npy")
    TP_unique = np.load("unique\\ROC_Curves\\bayes_TP.npy")
    FP_unique = np.load("unique\\ROC_Curves\\bayes_FP.npy")
    TP_non_unique = np.load("non_unique\\ROC_Curves\\bayes_TP.npy")
    FP_non_unique = np.load("non_unique\\ROC_Curves\\bayes_FP.npy")
    TP_unique_skin = np.load("unique_skin\\ROC_Curves\\bayes_TP.npy")
    FP_unique_skin = np.load("unique_skin\\ROC_Curves\\bayes_FP.npy")

    plotROCCurve(TP_unique_non_skin, FP_unique_non_skin,  '--', 1.5)
    plotROCCurve(TP_unique, FP_unique, '-', 1.5)
    plotROCCurve(TP_non_unique, FP_non_unique, '-', 2.5)
    plotROCCurve(TP_unique_skin, FP_unique_skin, '-.', 1.5)

    ax = plt.gca()
    ax.set_xticks(np.arange(0, 1.01, 0.05))
    ax.set_yticks(np.arange(0, 1.01, 0.05))
    ax.set_xlim(0.2, 0.601)
    ax.set_ylim(0.65, 1.0)

    plt.legend([u"Unique background", u"Unique", u"Classical method", u"Unique skin"], loc = 'lower right')


"""
Аппроксимация значений функции в требуемых точках.

Args:
    points: точки, в которых проводится аппроксимация.
    x: массив аргументов функции в известных точках.
    y: массив значений функции в известных точках.
    n: количество ближайших точек для аппроксимации.

Returns:
    Значения функции в точках аппроксимации.
"""
def approximate(points, x, y, n):
    result = np.zeros_like(points)

    for index in np.arange(len(points)):
        difference = np.abs(x - points[index])

        near = difference.argsort()[:n]
        near_x = x[near]
        near_y = y[near]

        p = np.polyfit(near_x, near_y, 2)
        result[index] = np.polyval(p, points[index])

    return result


"""
Построение процентного соотношения TPR между классификаторами Байеса,
обученными предложенными методами и классическим методом.

Args:
    n: количество ближайших точек для аппроксимации.
"""
def plotRatioTPR(n):
    TP_unique_non_skin = np.load("unique_non_skin\\ROC_Curves\\bayes_TP.npy")
    FP_unique_non_skin = np.load("unique_non_skin\\ROC_Curves\\bayes_FP.npy")
    TP_unique = np.load("unique\\ROC_Curves\\bayes_TP.npy")
    FP_unique = np.load("unique\\ROC_Curves\\bayes_FP.npy")
    TP_non_unique = np.load("non_unique\\ROC_Curves\\bayes_TP.npy")
    FP_non_unique = np.load("non_unique\\ROC_Curves\\bayes_FP.npy")
    TP_unique_skin = np.load("unique_skin\\ROC_Curves\\bayes_TP.npy")
    FP_unique_skin = np.load("unique_skin\\ROC_Curves\\bayes_FP.npy")

    TP_unique_non_skin = approximate(np.arange(0.2, 0.6, 0.01), FP_unique_non_skin, TP_unique_non_skin, n)
    TP_unique = approximate(np.arange(0.2, 0.6, 0.01), FP_unique, TP_unique, n)
    TP_non_unique = approximate(np.arange(0.2, 0.6, 0.01), FP_non_unique, TP_non_unique, n)
    TP_unique_skin = approximate(np.arange(0.2, 0.6, 0.01), FP_unique_skin, TP_unique_skin, n)

    plotROCCurve((TP_unique_non_skin / TP_non_unique - 1) * 100, np.arange(0.2, 0.6, 0.01),  '--', 1.5)
    plotROCCurve((TP_unique / TP_non_unique - 1) * 100, np.arange(0.2, 0.6, 0.01), '-', 1.5)
    plotROCCurve((TP_unique_skin / TP_non_unique - 1) * 100, np.arange(0.2, 0.6, 0.01), '-.', 1.5)

    ax = plt.gca()
    ax.set_xticks(np.arange(0, 2.01, 0.05))
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.set_xlim(0.2, 0.6)
    ax.set_ylim(-5, 5)
    ax.set(xlabel = 'FPR', ylabel = r'$(TPR - TPR_{classic}) / TPR_{classic}$, %')
    plt.legend([u"Unique background", u"Unique", u"Unique skin"], loc = 'lower right')


"""
Построение процентного соотношения FPR между классификаторами Байеса,
обученными предложенными методами и классическим методом.

Args:
    n: количество ближайших точек для аппроксимации.
"""
def plotRatioFPR(n):
    TP_unique_non_skin = np.load("unique_non_skin\\ROC_Curves\\bayes_TP.npy")
    FP_unique_non_skin = np.load("unique_non_skin\\ROC_Curves\\bayes_FP.npy")
    TP_unique = np.load("unique\\ROC_Curves\\bayes_TP.npy")
    FP_unique = np.load("unique\\ROC_Curves\\bayes_FP.npy")
    TP_non_unique = np.load("non_unique\\ROC_Curves\\bayes_TP.npy")
    FP_non_unique = np.load("non_unique\\ROC_Curves\\bayes_FP.npy")
    TP_unique_skin = np.load("unique_skin\\ROC_Curves\\bayes_TP.npy")
    FP_unique_skin = np.load("unique_skin\\ROC_Curves\\bayes_FP.npy")

    FP_unique_non_skin = approximate(np.arange(0.65, 1.0, 0.01), TP_unique_non_skin, FP_unique_non_skin, n)
    FP_unique = approximate(np.arange(0.65, 1.0, 0.01), TP_unique, FP_unique, n)
    FP_non_unique = approximate(np.arange(0.65, 1.0, 0.01), TP_non_unique, FP_non_unique, n)
    FP_unique_skin = approximate(np.arange(0.65, 1.0, 0.01), TP_unique_skin, FP_unique_skin, n)

    plotROCCurve(np.arange(0.65, 1.0, 0.01), (FP_unique_non_skin / FP_non_unique - 1) * 100,  '--', 1.5)
    plotROCCurve(np.arange(0.65, 1.0, 0.01), (FP_unique / FP_non_unique - 1) * 100, '-', 1.5)
    plotROCCurve(np.arange(0.65, 1.0, 0.01), (FP_unique_skin / FP_non_unique - 1) * 100, '-.', 1.5)

    ax = plt.gca()
    ax.set_xticks(np.arange(-15, 6, 2.5))
    ax.set_yticks(np.arange(0, 2.01, 0.05))
    ax.set_xlim(-15, 5.1)
    ax.set_ylim(0.65, 1.0)
    ax.set(xlabel = r'$(FPR - FPR_{classic}) / FPR_{classic}$, %', ylabel = 'TPR')

    plt.legend([u"Unique background", u"Unique", u"Unique skin"], loc = 'lower left')
