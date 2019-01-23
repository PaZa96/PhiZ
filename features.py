from math import hypot
import math
import numpy as np

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

def s_triangle(point_a, point_b, point_c):

    a = face_features_length(point_a, point_b)
    b = face_features_length(point_b, point_c)
    c = face_features_length(point_c, point_a)

    p = (a + b + c) / 2

    s = math.sqrt(p*(p-a)*(p-b)*(p-c))

    return s

def draw_face_line(point_one, point_two, overlay, pts): 

    for l in range(0, len(pts)):
        ptA = tuple(shape[point_one])
        ptB = tuple(shape[point_two])
        cv2.line(overlay, ptA, ptB, (255, 0, 255), 1)

def line_angle(shape, pointA, pointB, pointC):

    points = np.array([[shape[pointA][0],shape[pointA][1]], [shape[pointB][0],shape[pointB][1]], [int(shape[pointC][0] * 1.1),shape[pointC][1]]]) 

    A = points[1] - points[0] 
    B = points[2] - points[1] 
    C = points[0] - points[2] 

    angles = [] 
    for e1, e2 in ((A, -B), (B, -C), (C, -A)): 
        num = np.dot(e1, e2) 
        denom = np.linalg.norm(e1) * np.linalg.norm(e2) 
        angles.append(np.arccos(num/denom) * 180/np.pi) 
    
        print("Angle")
    print (angles) 
    print (sum(angles)) 
