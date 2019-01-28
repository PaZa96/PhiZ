from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image
import cv2
import argparse
import dlib
import imutils
import os
import phiz
import uuid
 
root = Tk()
root.title("PhiZ")
root.geometry("1030x700")

imagepath="../PhiZ/images/noimage.png"
shape = {}
character_type = {"Холерик":1, "Флегматик":1, "Сангвиник":1, "Меланхолик":1}

def character():
    global character_type
    return character_type

def upload_image():
    global imagepath
    file_name = fd.askopenfilename()
    imagepath = file_name
    picture = Image.open(imagepath)
    picture = picture.resize((450, 560))
    img = ImageTk.PhotoImage(picture)
    panel = Label(root, image = img)
    panel.place(x=30, y=20)
    root.mainloop()

def analisis_image():
    # инициализировать детектор лица dlib (на основе HOG), а затем создать
    # предиктор ориентир лиц
    output_text = []
    detector = dlib.get_frontal_face_detector()
    str = uuid.uuid4().hex 
    predictor = dlib.shape_predictor('../PhiZ/shape_predictor_68_face_landmarks.dat')

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(imagepath)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # обнаружение лиц на изображении в градациях серого
    rects = detector(gray, 1)

        # цикл по распознаванию лиц
    for (i, rect) in enumerate(rects):
   
        # определить ориентиры лица для области лица, затем
        # преобразовать координаты ориентира (x, y) в массив NumPy
        gshape = predictor(gray, rect)
        gshape = phiz.shape_to_numpy_array(gshape)
        shape = gshape
    
    output = phiz.visualize_facial_landmarks(image, shape)
    output_image = '../PhiZ/analysed_images/' + str + '.jpg'
    cv2.imwrite(output_image, output)

    picture = Image.open(output_image)
    picture = picture.resize((450, 560))
    img = ImageTk.PhotoImage(picture)
    panel = Label(root, image = img)
    panel.place(x=30, y=20)

    output_text = output_text + phiz.eyebrow(shape)
    output_text = output_text + phiz.eye(shape)
    output_text = output_text + phiz.nose(shape)
    output_text = output_text + phiz.mouth(shape)
    output_text = output_text + phiz.chip(shape)

    myString = '\n'.join(output_text) 

    label1= Label(root, text="Характеристика", justify = CENTER, wraplength = 480, font = "Arial 16", width = 40)
    label1.place(x=533, y=10)

    label = Label(root,    
                 wraplength = 480,
                 justify = LEFT,
                 font="14", 
                 text=myString)
    label.place(x=533, y=50)

    root.mainloop()


img = ImageTk.PhotoImage(Image.open(imagepath))
panel = Label(root, image = img)
panel.place(x=120, y=150)

 
btn = Button(text="Загрузить изображение",          # текст кнопки 
             background="#555",     # фоновый цвет кнопки
             foreground="#ccc",     # цвет текста
             padx="1",             # отступ от границ до содержимого по горизонтали
             pady="8",              # отступ от границ до содержимого по вертикали
             font="16",              # высота шрифта
             command=upload_image
             )
btn.place(x=50, y=620)

btn1 = Button(text="Анализировать",          # текст кнопки 
             background="#555",     # фоновый цвет кнопки
             foreground="#ccc",     # цвет текста
             padx="20",             # отступ от границ до содержимого по горизонтали
             pady="8",              # отступ от границ до содержимого по вертикали
             font="16",              # высота шрифта
             command=analisis_image
             )
btn1.place(x=300, y=620)

separator = Button(text="",          # текст кнопки 
             background="#555",     # фоновый цвет кнопки
             foreground="#ccc",     # цвет текста
             padx="0",             # отступ от границ до содержимого по горизонтали
             pady="700",              # отступ от границ до содержимого по вертикали
             font="16",             # высота шрифта
             state=DISABLED
             )
separator.pack()



root.mainloop()