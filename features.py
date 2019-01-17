from math import hypot
import math

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


