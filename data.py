#coding=utf-8
import numpy as np
import os
import cv2
from numpy.random import randint


"""
Сравнение пикселей трёхканального изображения в лексикографическом порядке.

Args:
    first: первый пиксель.
    second: второй пиксель.

Returns:
    1, если first > second.
    -1, если first < second.
    0, если first = second.
"""
def cmpFun(firts, second):
    if firts[0] > second[0]:
        return 1
    elif firts[0] < second[0]:
        return -1
    else:
        if firts[1] > second[1]:
            return 1
        elif firts[1] < second[1]:
            return -1
        else:
            if firts[2] > second[2]:
                return 1
            elif firts[2] < second[2]:
                return -1
            else:
                return 0

"""
Удаление из массива данных повторяющихся точек.

Args:
    data: массив данных.

Returns:
    Массив данных с уникальными точками.
"""
def unifyData(data):
    data = data.tolist()
    data.sort(cmp = cmpFun)

    result = [data[0]]

    for i in np.arange(1, len(data)):
        if data[i-1] == data[i]:
            continue

        result.append(data[i])

    return np.array(result, dtype = 'uint8')


"""
Сбор образцов точек кожи и фона с изображения.

Args:
    image: входное цветное изображение.
    ground_truth: бинарное размеченное изображение.
    fg_label: метка объектов на изобржении или -1 если объекты отмечены не уникальными метками.
    bg_label: метка фона на изображении или -1 если фон не имеет укникальной метки.

Returns:
    Массив образцов точек кожи.
    Массив образцов точек фона.
"""
def getDataFromImage(image, ground_truth, fg_label = -1, bg_label = -1):
    skin = np.empty([0, 3], dtype = np.uint8)
    non_skin = np.empty([0, 3], dtype = np.uint8)
    rows, cols, channels = image.shape

    for y in np.arange(rows):
        for x in np.arange(cols):
            current = image[y,x,:]
            if (ground_truth[y, x] == fg_label) or (fg_label == -1 and ground_truth[y, x] != bg_label):
                skin = np.vstack((skin, current))
            elif (ground_truth[y, x] == bg_label) or (bg_label == -1 and ground_truth[y, x] != fg_label):
                non_skin = np.vstack((non_skin, current))
            else:
                print "Error at pixel ", y, x
                continue

    skin = unifyData(skin);
    non_skin = unifyData(non_skin);

    return skin, non_skin


"""
Сбор данных с изображений из каталога.

Args:
    image_dir: путь к каталогу с изображениями.
    ground_truth_dir: путь к каталогу с размеченными бинарными изображениями.
    ground_truth_ext: расширение размеченных изображений
    fg_label: метка объектов на изобржении или -1 если объекты отмечены не уникальными метками.
    bg_label: метка фона на изображении или -1 если фон не имеет укникальной метки.

Returns:
    Массив образцов точек кожи.
    Массив образцов точек фона.
"""
def getDataFromDirectory(image_dir, ground_truth_dir, ground_truth_ext, fg_label, bg_label):
    skin = np.empty([0, 3], dtype = np.uint8)
    non_skin = np.empty([0, 3], dtype = np.uint8)

    images = os.listdir(image_dir)
    images = filter(lambda x: x.endswith('.jpg') or x.endswith('.JPG') or x.endswith('.jpeg') or x.endswith('.jpeg'), images)

    for file in images:
        print "Current file:", file
        ext_index = file.find('.')
        name = file[0:ext_index]
        gt_file = ground_truth_dir + name + ground_truth_ext
        image_file = image_dir + file
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        gt_image = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)

        if image == None or gt_image == None:
            print 'Can not find file', file
            continue

        current_skin, current_non_skin = getDataFromImage(image, gt_image, fg_label, bg_label)
        skin = np.vstack((skin, current_skin))
        non_skin = np.vstack((non_skin, current_non_skin))

    return skin, non_skin


"""
Сбор данных из текстового файла UCI датасета.

Args:
    path: путь к каталогу с файлом датасета.

Returns:
    Массив образцов точек кожи.
    Массив образцов точек фона.
"""
def getDataFromUCI(path):
    file = path + '\\Skin_NonSkin(BGR).txt'
    fid = open(file, 'r')
    skin = np.empty([0,3], dtype = np.uint8)
    non_skin = np.empty([0,3], dtype = np.uint8)

    for line in fid:
        values = line.split()
        current = np.array([int(values[0]), int(values[1]), int(values[2])], dtype = np.uint8)
        if values[3] == '1':
            skin = np.vstack((skin, current))
        else:
            non_skin = np.vstack((non_skin, current))

    fid.close()
    return skin, non_skin


"""
Сбор данных из датасета FaceDataset.

Args:
    path: путь к каталогу с датасетом.

Returns:
    Массив образцов точек кожи.
    Массив образцов точек фона.
"""
def getDataFromFaceDataset(path):
    image_dir = path + '\\Pratheepan_Dataset\\FacePhoto\\'
    ground_truth_dir = path + '\\Ground_Truth\\GroundT_FacePhoto\\'
    skin_face, non_skin_face = getDataFromDirectory(image_dir, ground_truth_dir, '.png', 255, 0)

    image_dir = path + '\\Pratheepan_Dataset\\FamilyPhoto\\'
    ground_truth_dir = path + '\\Ground_Truth\\GroundT_FamilyPhoto\\'
    skin_family, non_skin_family = getDataFromDirectory(image_dir, ground_truth_dir, '.png', 255, 0)

    skin = np.vstack((skin_face, skin_family))
    non_skin = np.vstack((non_skin_face, non_skin_family))

    return skin, non_skin


"""
Сбор данных из датасета IBTD.

Args:
    path: путь к каталогу с датасетом.

Returns:
    Массив образцов точек кожи.
    Массив образцов точек фона.
"""
def getDataFromIBTD(path):
    image_dir = path + '\\'
    ground_truth_dir = path + '\\Mask\\'

    skin, non_skin = getDataFromDirectory(image_dir, ground_truth_dir, '.bmp', -1, 255)
    return skin, non_skin


"""
Сбор данных из датасета Hands Gesture Recognition.

Args:
    path: путь к каталогу с датасетом.

Returns:
    Массив образцов точек кожи.
    Массив образцов точек фона.
"""
def getDataFromHGR(path):
    skin = np.empty([0, 3], dtype = np.uint8)
    non_skin = np.empty([0, 3], dtype = np.uint8)

    for set_name in ['HGR1', 'HGR2A', 'HGR2B']:
        image_dir = path + '\\' + set_name + '\\original_images\\'
        ground_truth_dir = path + '\\' + set_name + '\\skin_masks\\'
        skin_tmp, non_skin_tmp = getDataFromDirectory(image_dir, ground_truth_dir, '.bmp', 0, 255)
        skin = np.vstack((skin, skin_tmp))
        non_skin = np.vstack((non_skin, non_skin_tmp))

    return skin, non_skin

"""
Сбор данных из всех доступных источников.

Returns:
    Массив образцов точек кожи.
    Массив образцов точек фона.
"""
def collectTrainingData(path):
    try:
        skin = np.load(path + '\\UCI_skin.npy')
        non_skin = np.load(path + '\\UCI_non_skin.npy')
    except:
        skin, non_skin = getDataFromUCI('D:\\Work\\Datasets\\UCI')
        np.save(path + '\\UCI_skin.npy', skin)
        np.save(path + '\\UCI_non_skin.npy', non_skin)

    try:
        skin_tmp = np.load(path + '\\HGR_skin.npy')
        non_skin_tmp = np.load(path + '\\HGR_non_skin.npy')
    except:
        skin_tmp, non_skin_tmp = getDataFromHGR('D:\\Work\\Datasets\\HandGestureRecognition')
        np.save(path + '\\HGR_skin.npy', skin_tmp)
        np.save(path + '\\HGR_non_skin.npy', non_skin_tmp)

    skin = np.vstack((skin, skin_tmp))
    non_skin = np.vstack((non_skin, non_skin_tmp))

    try:
        skin_tmp = np.load(path + '\\IBTD_skin.npy')
        non_skin_tmp = np.load(path + '\\IBTD_non_skin.npy')
    except:
        skin_tmp, non_skin_tmp = getDataFromIBTD('D:\\Work\\Datasets\\ibtd')
        np.save(path + '\\IBTD_skin.npy', skin_tmp)
        np.save(path + '\\IBTD_non_skin.npy', non_skin_tmp)

    skin = np.vstack((skin, skin_tmp))
    non_skin = np.vstack((non_skin, non_skin_tmp))

    try:
        skin_tmp = np.load(path + '\\Face_skin.npy')
        non_skin_tmp = np.load(path + '\\Face_non_skin.npy')
    except:
        skin_tmp, non_skin_tmp = getDataFromFaceDataset('D:\\Work\\Datasets\\FaceDataset')
        np.save(path + '\\Face_skin.npy', skin_tmp)
        np.save(path + '\\Face_non_skin.npy', non_skin_tmp)

    skin = np.vstack((skin, skin_tmp))
    non_skin = np.vstack((non_skin, non_skin_tmp))

    print 'Skin data contains ', skin.shape[0], ' pixels.'
    print 'Non skin data contains ', non_skin.shape[0], ' pixels.'
    return skin, non_skin


"""
Возвращает массивы всех доступных данных.
"""
def getAllData():
    skin, non_skin = collectTrainingData()
    return skin, non_skin


"""
Разбиение случайным образом всех доступных данных на тестовую выборку
и обучающую выборку в соотношении 20% и 80% соответственно.
Тестовая выборка сохраняется в текущем каталоге в файлы test_skin_set.npy и
test_non_skin_set.npy, а обучающая выборка в training_skin_set.npy
и training_non_skin_set.npy.
"""
def splitData(skin, non_skin):
    skin_length = skin.shape[0]
    test_length = int(skin_length * 0.2)
    np.random.shuffle(skin)
    test_skin_set = skin[0:test_length, :]
    training_skin_set = skin[test_length:skin_length, :]
    np.save("test_skin_set.npy", test_skin_set)
    np.save("training_skin_set.npy", training_skin_set)

    non_skin_length = non_skin.shape[0]
    test_length = int(non_skin_length * 0.2)
    np.random.shuffle(non_skin)
    test_non_skin_set = non_skin[0:test_length, :]
    training_non_skin_set = non_skin[test_length:non_skin_length, :]
    np.save("test_non_skin_set.npy", test_non_skin_set)
    np.save("training_non_skin_set.npy", training_non_skin_set)


"""
Возвращает массивы обучающей выборки.
"""
def getTrainingSet():
    skin = np.load('data\\training_skin_set.npy')
    non_skin = np.load('data\\training_non_skin_set.npy')
    return skin, non_skin


"""
Возвращает массивы тестовой выборки.
"""
def getTestSet():
    skin = np.load('data\\test_skin_set.npy')
    skin = unifyData(skin)
    non_skin = np.load('data\\test_non_skin_set.npy')
    non_skin = unifyData(non_skin)
    return skin, non_skin


"""
Возвращает массивы обучающей выборки в фомрате YCrCb.
"""
def getTrainingSetYCrCb():
    skin, non_skin = getTrainingSet()

    num_samples = skin.shape[0]
    skin.resize(num_samples, 1, 3)
    skin_YCrCb = cv2.cvtColor(skin, cv2.COLOR_BGR2YCrCb)
    skin_YCrCb.resize(num_samples, 3)

    num_samples = non_skin.shape[0]
    non_skin.resize(num_samples, 1, 3)
    non_skin_YCrCb = cv2.cvtColor(non_skin, cv2.COLOR_BGR2YCrCb)
    non_skin_YCrCb.resize(num_samples, 3)

    return skin_YCrCb, non_skin_YCrCb


"""
Возвращает массивы тестовой выборки в формате YCrCb.
"""
def getTestSetYCrCb():
    skin, non_skin = getTestSet()

    num_samples = skin.shape[0]
    skin.resize(num_samples, 1, 3)
    skin_YCrCb = cv2.cvtColor(skin, cv2.COLOR_BGR2YCrCb)

    num_samples = non_skin.shape[0]
    non_skin.resize(num_samples, 1, 3)
    non_skin_YCrCb = cv2.cvtColor(non_skin, cv2.COLOR_BGR2YCrCb)

    return skin_YCrCb, non_skin_YCrCb


"""
Разбиение и унификация массива пикселей.
Части массива сохраняются в подкаталоге Temp.

Args:
    array: массив пикселей.

Returns:
    Количество сохранённых частей массива.
"""
def splitArray(array):
    length = array.shape[0]

    step = 1000000
    num_split = length / step

    for i in np.arange(num_split):
        temp_array = array[i*step:(i+1)*step, :]
        temp_array = unifyData(temp_array)
        np.save("Temp/" + str(i) + ".npy", temp_array)

    if (length % step) != 0:
        temp_array = array[num_split*step:length, :]
        temp_array = unifyData(temp_array)
        np.save("Temp/" + str(num_split) + ".npy", temp_array)

    return num_split


"""
Объединение частей массива из подкаталога Temp.

Args:
    path: путь к каталогу с датасетом.

Returns:
    Массив образцов точек кожи.
    Массив образцов точек фона.
"""
def collectArray(num_split):
    array = np.empty([0,3], dtype = np.uint8)

    for i in np.arange(num_split+1):
        temp_array = np.load("Temp/" + str(i) + ".npy")
        array = np.vstack((array, temp_array))

    array = unifyData(array)

    return array


if __name__ == '__main__':
    # Унификация точек кожи
    array = np.load('training_skin_set.npy')
    print 'Before unify skin data contain:', array.shape[0], 'points'
    num_split = splitArray(array)
    array = []
    array = collectArray(num_split)
    print 'After unify skin data contain:', array.shape[0], 'points'
    np.save('training_skin_set.npy', array)

    # Унификация точек фона
    array = np.load('training_non_skin_set.npy')
    print 'Before unify non_skin data contain:', array.shape[0], 'points'
    num_split = splitArray(array)
    array = []
    array = collectArray(num_split)
    print 'After unify non_skin data contain:', array.shape[0], 'points'
    np.save('training_non_skin_set.npy', array)
