# import the necessary packages
from collections import OrderedDict
from math import hypot
import math
import numpy as np
import cv2
import argparse
import dlib
import imutils
import os
import json

facial_features_cordinates = {}


# определить словарь, который отображает индексы лица
# ориентиры для определенных областей лица
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (22, 27)),
    ("Left_Eyebrow", (17, 22)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 36)),
    ("Jaw", (0, 17))
])

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
#	help="path to facial landmark predictor")
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())


def shape_to_numpy_array(shape, dtype="int"):
    
    # инициализировать список (x, y) -координат
    coordinates = np.zeros((68, 2), dtype=dtype)

    
    # перебрать 68 лицевых ориентиров и конвертировать их
    # для двух кортежей (x, y) -координат
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    
    # вернуть список (x, y) -координат
    return coordinates

def face_features_length(x, y): 

    distance = hypot(x[0] - y[0], x[1] - y[1])
    print(distance)

    return distance

def coordinates_center_line(x, y):

    coordinates = []

    dx = int((x[0] + y[0])/2)
    dy = int((x[1] + y[1])/2)
    
    coordinates.append(dx)
    coordinates.append(dy)

    print(coordinates)

    return coordinates

def s_quadrilateral(a, b, c, d):

    p = (a + b + c + d) / 2

    s = math.sqrt((p-a)*(p-b)*(p-c)*(p-d))

    return s

def draw_face_line(point_one, point_two, overlay, pts): 

    for l in range(0, len(pts)):
        ptA = tuple(shape[point_one])
        ptB = tuple(shape[point_two])
        cv2.line(overlay, ptA, ptB, (255, 0, 255), 1)


def draw_face_features(facial_features_name, overlay): 
     # цикл по областям лицевой ориентир индивидуально
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
     
        # захватить (x, y) -координаты, связанные с
        # лицо ориентир
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

       
        
    
        # так как линия челюсти не является закрытой лицевой областью,
        # просто рисуем линии между (x, y) -координатами
        if name == "Right_Eye" or name == "Left_Eye":
            for l in range(0, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, (50, 255, 50), 2)

            for l in range(0, len(pts)):
                ptA = tuple(pts[l])
                cv2.circle(overlay, ptA, 0, (19,19,19), 3)
                

        elif name == "Nose":

            coordinates_point_nose_rigth = []
            coordinates_point_nose_left = []
            cn = coordinates_center_line(shape[30],coordinates_center_line(shape[31], shape[35])) 

            coordinates_point_nose_rigth.append(int((round(shape[35][0] - shape[34][0])) + shape[35][0]))
            coordinates_point_nose_rigth.append(cn[1])
            coordinates_point_nose_left.append(int(round(shape[31][0] - (shape[32][0] - shape[31][0]))))
            coordinates_point_nose_left.append(cn[1])

            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                ptC = tuple(pts[3])
                ptD = tuple(pts[8])

                cv2.line(overlay, ptA, ptB, (50, 255, 50), 2)
                cv2.line(overlay, ptC, ptD, (50, 255, 50), 2)

            for l in range(0, len(pts)):
                ptA = tuple(pts[l])
                
                cv2.circle(overlay, ptA, 0, (19,19,19), 3)
                cv2.circle(overlay, tuple(coordinates_point_nose_rigth), 0, (19,19,19), 3)
                cv2.circle(overlay, tuple(coordinates_point_nose_left), 0, (19,19,19), 3)

        elif name == "Mouth":
            for l in range(0, len(pts)):
                ptA = tuple(pts[l-1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, (50, 255, 50), 2)

            for l in range(0, len(pts)):
                ptA = tuple(pts[l])
                cv2.circle(overlay, ptA, 0, (19,19,19), 3)


        else :
            for l in range(1, len(pts)):
                ptA = tuple(pts[l-1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, (50, 255, 50), 2)

            for l in range(0, len(pts)):
                ptA = tuple(pts[l])
                cv2.circle(overlay, ptA, 0, (19,19,19), 3)

                center_eyebrow_point = coordinates_center_line(shape[17],shape[21])

                cv2.circle(overlay, tuple(center_eyebrow_point), 0, (19,19,19), 3)

    draw_face_line(51, 57, overlay, pts)
    draw_face_line(21, 22, overlay, pts)
    draw_face_line(39, 42, overlay, pts)


# форма лица -----------------------------------------------------

    draw_face_line(0, 16, overlay, pts)
    draw_face_line(2, 14, overlay, pts)
    draw_face_line(3, 13, overlay, pts)
    draw_face_line(4, 12, overlay, pts)
    draw_face_line(17, 21, overlay, pts)


   

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
   
    # создать две копии входного изображения - одну для
    # оверлей и один для окончательного выходного изображения
    overlay = image.copy()
    output = image.copy()


    # если список цветов Нет, инициализируйте его уникальным
    # цвет для каждого региона лица ориентир

 
        
    draw_face_features("Jaw", overlay)
    draw_face_features("Nose", overlay)
    draw_face_features("Mouth", overlay)
    draw_face_features("Right_Eyebrow", overlay)
    draw_face_features("Left_Eyebrow", overlay)
    draw_face_features("Right_Eye", overlay)
    draw_face_features("Left_Eye", overlay, )
        
     
    
    # применить прозрачный оверлей
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    
    # вернуть выводимое изображение
    print(facial_features_cordinates)

    return output




# инициализировать детектор лица dlib (на основе HOG), а затем создать
# предиктор ориентир лица


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('C:/Users/Zhavoronok/Desktop/Detect-Facial-Features-master/shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('C:/Users/Zhavoronok/Desktop/Detect-Facial-Features-master/images/image9.jpg')
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# обнаружение лиц на изображении в градациях серого
rects = detector(gray, 1)


def eyebrow(shape, left_point, right_point):

    print("Eyebrow horizontal length")
    eyebrow_width = face_features_length(shape[left_point], shape[right_point]) #вычисляем длину брови 
    print(eyebrow_width)

    center_eyebrow_point = coordinates_center_line(shape[left_point],shape[right_point]) # находим центр отрезка между началом и концом брови

    point = left_point + 1
    length_between_points = []

    while point < right_point: # находим длину между центральной точкой и точками на брови
        length = face_features_length(shape[point], center_eyebrow_point)
        length_between_points.append(length)
        point = point + 1

    left_section = length_between_points[0] 
    right_section = length_between_points[2]

    left_coefficient = (left_section / length_between_points[1]) 
    right_coefficient = (right_section / length_between_points[1]) 

    print("eyebrow coeff")
    print( right_section / left_section  )
    





def nose(shape):

    print("Nose horizontal length")

    coordinates_point_nose_rigth = []
    coordinates_point_nose_left = []
    cn = coordinates_center_line(shape[30],coordinates_center_line(shape[31], shape[35])) 

    coordinates_point_nose_rigth.append(int((round(shape[35][0] - shape[34][0])) + shape[35][0]))
    coordinates_point_nose_rigth.append(cn[1])
    coordinates_point_nose_left.append(int(round(shape[31][0] - (shape[32][0] - shape[31][0]))))
    coordinates_point_nose_left.append(cn[1])

    nose_length_horizontal = face_features_length(coordinates_point_nose_left, coordinates_point_nose_rigth)
    print("---------------------------")
    print("length between nose and center of mouth")
    point_center_mouth = face_features_length(shape[33], coordinates_center_line(shape[62], shape[66]))
    print("---------------------------")

    print("Calculated Nose width coefficient")

    nose_coefficient = nose_length_horizontal / point_center_mouth

    nose_coefficient = nose_coefficient - (nose_coefficient * 0.09) 

    print(nose_coefficient)

    

def mouth(shape):

    # определяем размеры рта и губ

    print("Mouth horizontal length")
    mouth_lenght_horizontal = face_features_length(shape[48], shape[54])
    print("---------------------------")
    print("Mouth vertical length")
    mouth_lenght_vertical = face_features_length(shape[51], shape[57])
    print("---------------------------")
    print("Lips1 vertical length")
    lip1_vertical_length  = face_features_length(shape[51], shape[62])
    print("---------------------------")
    print("Lips2 vertical length")
    lip2_vertical_length = face_features_length(shape[66], shape[57])
    print("---------------------------")

    # форма лица -----------------------------------------------------
    cw1 = face_features_length(shape[2], shape[14])
    cw2 = face_features_length(shape[4], shape[12])
    print("---------------------------")

    dcw = (cw1 + cw2) / 2

    sdf = dcw / mouth_lenght_horizontal

    print("ertjyjtyjkyrtuktk")
    print(sdf)

# цикл по распознаванию лиц
for (i, rect) in enumerate(rects):
   
    # определить ориентиры лица для области лица, затем
    # преобразовать координаты ориентира (x, y) в массив NumPy
    shape = predictor(gray, rect)
    shape = shape_to_numpy_array(shape)

# форма лица -----------------------------------------------------
print("face form")
face_features_length(shape[0], shape[16])
m1 = face_features_length(shape[2], shape[14])
face_features_length(shape[3], shape[13])
m2 = face_features_length(shape[4], shape[12])
print("---------------------------")
print("Eyebrow lenght")
face_features_length(shape[21], shape[22])
print("---------------------------")
print("Eye lenght")
face_features_length(shape[39], shape[42])
print("---------------------------")



min_lenght = 0
mouth_list = {}

#print("Губы")
#dm = (m1+m2) / 2  # среднее расстояние в районе губ
#delm = dm / 9 
#print(dm)

#delmm = delm
#it = 1

#while delmm <= dm:
#    mouth = abs(delmm - mouth_lenght)
#    mouth_list[it] = mouth
#    delmm += delm
#    it += 1





#print(min(mouth_list.values()))

#with open('face-features.json') as json_file:  
#    data = json.load(json_file)
#    for p in data['Face']['Mouth']:


#     #   if mouth_list[min(mouth_list.values())].keys() >= int(p['rate_max']) and mouth_list[min(mouth_list.values())].keys() <= int(p['rate_min']):

#     if min(mouth_list, key=mouth_list.get) <= int(p['rate_max']) and min(mouth_list, key=mouth_list.get) >= int(p['rate_min']):
#            print('Описание: ' + p['description'])

eyebrow(shape, 17,21)
nose(shape)
mouth(shape)


output = visualize_facial_landmarks(image, shape)
cv2.imshow("Image", output)
cv2.waitKey(0)
