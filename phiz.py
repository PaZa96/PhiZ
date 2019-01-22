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
import features

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
            #    cv2.line(overlay, ptA, ptB, (50, 255, 50), 2)

            for l in range(0, len(pts)):
                ptA = tuple(pts[l])
                cv2.circle(overlay, ptA, 0, (50, 255, 50), 3)
                

        elif name == "Nose":

            coordinates_point_nose_rigth = []
            coordinates_point_nose_left = []
            cn = features.coordinates_center_line(shape[30],features.coordinates_center_line(shape[31], shape[35])) 

            coordinates_point_nose_rigth.append(int((round(shape[35][0] - shape[34][0])) + shape[35][0]))
            coordinates_point_nose_rigth.append(cn[1])
            coordinates_point_nose_left.append(int(round(shape[31][0] - (shape[32][0] - shape[31][0]))))
            coordinates_point_nose_left.append(cn[1])

            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                ptC = tuple(pts[3])
                ptD = tuple(pts[8])

             #   cv2.line(overlay, ptA, ptB, (50, 255, 50), 2)
              #  cv2.line(overlay, ptC, ptD, (50, 255, 50), 2)

            for l in range(0, len(pts)):
                ptA = tuple(pts[l])
                
                cv2.circle(overlay, ptA, 0, (50, 255, 50), 3)
                cv2.circle(overlay, tuple(coordinates_point_nose_rigth), 0, (50, 255, 50), 3)
                cv2.circle(overlay, tuple(coordinates_point_nose_left), 0, (50, 255, 50), 3)

        elif name == "Mouth":
            for l in range(0, len(pts)):
                ptA = tuple(pts[l-1])
                ptB = tuple(pts[l])
               # cv2.line(overlay, ptA, ptB, (50, 255, 50), 2)

            for l in range(0, len(pts)):
                ptA = tuple(pts[l])
                cv2.circle(overlay, ptA, 0, (50, 255, 50), 3)



        else :
            for l in range(1, len(pts)):
                ptA = tuple(pts[l-1])
                ptB = tuple(pts[l])
                #cv2.line(overlay, ptA, ptB, (50, 255, 50), 2)

            for l in range(0, len(pts)):
                ptA = tuple(pts[l])
                cv2.circle(overlay, ptA, 0, (50, 255, 50), 3)

                center_eyebrow_point = features.coordinates_center_line(shape[17],shape[21])
                left_center_eyebrow_point = features.coordinates_center_line(shape[17],center_eyebrow_point)
                right_center_eyebrow_point = features.coordinates_center_line(center_eyebrow_point,shape[21])

                cv2.circle(overlay, tuple(center_eyebrow_point), 0, (50, 255, 50), 3)
                cv2.circle(overlay, tuple(left_center_eyebrow_point), 0, (50, 255, 50), 3)
                cv2.circle(overlay, tuple(right_center_eyebrow_point), 0, (50, 255, 50), 3)

   # draw_face_line(51, 57, overlay, pts)
   # draw_face_line(21, 22, overlay, pts)
    #draw_face_line(39, 42, overlay, pts)


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

predictor = dlib.shape_predictor('../PhiZ/shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('../PhiZ/images/image.jpg')
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# обнаружение лиц на изображении в градациях серого
rects = detector(gray, 1)


def height_eyebrow(shape, left_point, right_point, point_eye):

    center_eyebrow_point = features.coordinates_center_line(shape[left_point],shape[right_point]) # находим координаты центра отрезка между крайними точками правой брови
    height = features.face_features_length(center_eyebrow_point, shape[point_eye])

    return height

def height_eye(shape, left_point_up_eye, right_point_up_eye, left_point_down_eye, right_point_down_eye):

    center_eye_up_point = features.coordinates_center_line(shape[left_point_up_eye],shape[right_point_up_eye]) # находим координаты центра отрезка между крайними точками правой брови
    center_eye_down_point = features.coordinates_center_line(shape[left_point_down_eye],shape[right_point_down_eye])
    height_eye = features.face_features_length(center_eye_up_point, center_eye_down_point)

    return height_eye

def type_eyebrow(shape, l_left_point, l_right_point, r_left_point, r_right_point):
    print("Eyebrow horizontal length")
    l_eyebrow_width = features.face_features_length(shape[l_left_point], shape[l_right_point]) #вычисляем длину левой брови 
    r_eyebrow_width = features.face_features_length(shape[r_left_point], shape[r_right_point]) #вычисляем длину правой брови

    center_l_eyebrow_point = features.coordinates_center_line(shape[l_left_point],shape[l_right_point]) # находим координаты центра отрезка между крайними точками правой брови
    center_r_eyebrow_point = features.coordinates_center_line(shape[r_left_point],shape[r_right_point]) # находим координаты центра отрезка между крайними точками левой брови

    left_center_l_eyebrow_point = features.coordinates_center_line(shape[l_left_point],center_l_eyebrow_point)
    right_center_l_eyebrow_point = features.coordinates_center_line(center_l_eyebrow_point,shape[l_right_point])

    b1 = features.face_features_length(left_center_l_eyebrow_point, center_l_eyebrow_point)
    b2 = features.face_features_length(center_l_eyebrow_point, shape[l_left_point+2])

    b = b2 / b1

    left_center_r_eyebrow_point = features.coordinates_center_line(shape[r_left_point],center_r_eyebrow_point)
    right_center_r_eyebrow_point = features.coordinates_center_line(center_r_eyebrow_point,shape[r_right_point])

    l_left_triangle = features.s_triangle(shape[l_left_point], shape[l_left_point + 1], center_l_eyebrow_point)
    l_right_triangle = features.s_triangle(left_center_l_eyebrow_point, shape[l_left_point + 2], right_center_l_eyebrow_point)

    r_left_triangle = features.s_triangle(left_center_r_eyebrow_point, shape[r_right_point - 2], right_center_r_eyebrow_point)
    r_right_triangle = features.s_triangle(shape[r_right_point], shape[r_right_point - 1], center_r_eyebrow_point)

    s1 = l_left_triangle + r_right_triangle
    s2 = l_right_triangle + r_left_triangle

    s =  s1 / s2

    print(s)
    print(b)

    eyebrow_variables = [s,b]

    print("Eyebrow coefficient _________")
    return eyebrow_variables

def eyebrow(shape):

    eyebrow_type = type_eyebrow(shape, 17, 21, 22, 26)
    print("Left eyebrow coefficient")
    print(eyebrow_type)

    with open('face-features.json') as json_file:  
        data = json.load(json_file)
        for p in data['Face']['Eyebrows']['General']:
            if eyebrow_type[0] >= float(p['rate_min']) and eyebrow_type[0] <= float(p['rate_max']) and eyebrow_type[1] >= float(p['relation_min']) and eyebrow_type[1] <= float(p['relation_max']):
                print('Описание: ' + p['description'])
                break

    height_left_eyebrow = height_eyebrow(shape, 17,21,37)
    height_right_eyebrow = height_eyebrow(shape, 22,26,44)
            
    eyebrow_height = (height_left_eyebrow + height_right_eyebrow) / 2

    height_left_eye = height_eye(shape, 37,38,40,41)
    height_right_eye = height_eye(shape, 43,44,47,46)
            
    eye_height = (height_left_eye + height_right_eye) / 2

    with open('face-features.json') as json_file:  
        data = json.load(json_file)
        if eyebrow_height > eye_height:
            print(data['Face']['Eyebrows']['Height'][0]['description'])
        else:
            print(data['Face']['Eyebrows']['Height'][1]['description'])



def nose(shape):

    print("Nose horizontal length")

    coordinates_point_nose_rigth = []
    coordinates_point_nose_left = []
    cn = features.coordinates_center_line(shape[30],features.coordinates_center_line(shape[31], shape[35])) 

    coordinates_point_nose_rigth.append(int((round(shape[35][0] - shape[34][0])) + shape[35][0]))
    coordinates_point_nose_rigth.append(cn[1])
    coordinates_point_nose_left.append(int(round(shape[31][0] - (shape[32][0] - shape[31][0]))))
    coordinates_point_nose_left.append(cn[1])

    nose_length_horizontal = features.face_features_length(coordinates_point_nose_left, coordinates_point_nose_rigth)
    print("---------------------------")
    print("length between nose and center of mouth")
    point_center_mouth = features.face_features_length(shape[33], features.coordinates_center_line(shape[62], shape[66]))
    print("---------------------------")

    print("Calculated Nose width coefficient")

    nose_coefficient = nose_length_horizontal / point_center_mouth

    nose_coefficient = nose_coefficient - (nose_coefficient * 0.09) 

    print(nose_coefficient)


def chip(shape):
    cw1 = features.face_features_length(shape[0], shape[16])
    cw2 = features.face_features_length(shape[2], shape[14])
    cw3 = features.face_features_length(shape[4], shape[12])

    print('Chip')

    with open('face-features.json') as json_file:  
        data = json.load(json_file)
        for p in data['Face']['Chip']:
            if cw3 * 1.2 > cw1:
                print(data['Face']['Chip'][0]['description'])
                break
            elif (cw3 / cw1) < 0.7:
                print(data['Face']['Chip'][1]['description'])
                break

def mouth(shape):

    # определяем размеры рта и губ

    print("Mouth horizontal length")
    mouth_length_horizontal = features.face_features_length(shape[48], shape[54])
    print("---------------------------")
    print("Mouth vertical length")
    mouth_length_vertical = features.face_features_length(shape[51], shape[57])
    print("---------------------------")
    print("Lips1 vertical length")
    lip1_vertical_length  = features.face_features_length(shape[51], shape[62])
    print("---------------------------")
    print("Lips2 vertical length")
    lip2_vertical_length = features.face_features_length(shape[66], shape[57])
    print("---------------------------")
    length_nose_chin = features.face_features_length(shape[33], shape[8])

    length_nose_center_lip = features.face_features_length(shape[33], shape[62])
    length_chip_center_lip = features.face_features_length(shape[8], shape[66])
    print("length_nose_chin")
    print(length_chip_center_lip / length_nose_center_lip)
    # форма лица -----------------------------------------------------
    cw1 = features.face_features_length(shape[3], shape[13])
    cw2 = features.face_features_length(shape[4], shape[12])
    print("---------------------------")

    dcw = (cw1 + cw2) / 2
    sdf = dcw / mouth_length_horizontal

    min_length = 0
    mouth_list = {}

    print("Губы")
    dm = (cw1+cw2) / 2  # среднее расстояние в районе губ
    delm = dm / 9 
    print(dm)

    delmm = delm
    it = 1

    while delmm <= dm:
        mouth = abs(delmm - mouth_length_horizontal)
        mouth_list[it] = mouth
        delmm += delm
        it += 1

    print(min(mouth_list.values()))

    with open('face-features.json') as json_file:  
        data = json.load(json_file)
        for p in data['Face']['Mouth']['General']:


            if min(mouth_list, key=mouth_list.get) <= int(p['rate_max']) and min(mouth_list, key=mouth_list.get) >= int(p['rate_min']):
                print('Описание: ' + p['description'])

        if  lip1_vertical_length > lip2_vertical_length * 1.3:
            print(data['Face']['Mouth']['Height_lip'][0]['description'])
        if  lip2_vertical_length > lip1_vertical_length * 1.3:
            print(data['Face']['Mouth']['Height_lip'][1]['description'])
        if  (length_nose_center_lip) < (length_nose_chin / 3) * 0.95:
            print(data['Face']['Mouth']['Height_lip_nose'][0]['description'])
        else:
             if  (length_nose_center_lip) > (length_nose_chin / 3):
                   print(data['Face']['Mouth']['Height_lip_nose'][1]['description'])
        if  (mouth_length_vertical / mouth_length_horizontal) < 0.2:
            print(data['Face']['Mouth']['Height_lip_nose'][2]['description'])
        if  (mouth_length_vertical / mouth_length_horizontal) > 0.4:
            print(data['Face']['Mouth']['Height_lip_nose'][3]['description'])

# цикл по распознаванию лиц
for (i, rect) in enumerate(rects):
   
    # определить ориентиры лица для области лица, затем
    # преобразовать координаты ориентира (x, y) в массив NumPy
    shape = predictor(gray, rect)
    shape = shape_to_numpy_array(shape)

# форма лица -----------------------------------------------------
print("face form")
features.face_features_length(shape[0], shape[16])
m1 = features.face_features_length(shape[2], shape[14])
features.face_features_length(shape[3], shape[13])
m2 = features.face_features_length(shape[4], shape[12])
print("---------------------------")
print("Eyebrow length")
features.face_features_length(shape[21], shape[22])
print("---------------------------")
print("Eye length")
features.face_features_length(shape[39], shape[42])
print("---------------------------")





eyebrow(shape)
nose(shape)
mouth(shape)
chip(shape)

output = visualize_facial_landmarks(image, shape)
cv2.imshow("Image", output)
cv2.waitKey(0)
