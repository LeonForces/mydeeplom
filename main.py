import cv2
import numpy as np
from sort import Sort
import psycopg2
import tkinter as tk
from tkinter import messagebox

# Функция для загрузки модели YOLO
def load_yolo_model():
    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Функция для загрузки имен классов
def load_classes():
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Функция для выполнения детекции с помощью YOLO
def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Порог детекции
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    result = []
    for i in range(len(boxes)):
        if i in indexes:
            box = boxes[i]
            result.append((*box, confidences[i], class_ids[i]))
    return result

# Функция для подключения к базе данных
def connect_db():
    try:
        conn = psycopg2.connect(
            dbname="deeplom",
            user="postgres",
            password="abubakir",
            host="localhost",
            port="5432"
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Ошибка подключения к базе данных: {e}")
        return None

# Функция для записи координат в базу данных
def insert_coordinates(conn, obj_id, frame_number, x, y):
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO object_tracking (obj_id, frame, x, y) VALUES (%s, %s, %s, %s)",
                (obj_id, frame_number, x, y)
            )
        conn.commit()
        #print(f"Inserted coordinates for object {obj_id} at frame {frame_number}: ({x}, {y})")
    except Exception as e:
        print(f"Ошибка записи координат в базу данных: {e}")

def show_alert():
    root = tk.Tk()
    root.withdraw()  # Скрыть главное окно
    messagebox.showinfo("Обнаружение объекта", "Человек обнаружен!")
    root.destroy()

# Инициализация трекера SORT
tracker = Sort()

# Загрузка видеопотока
cap = cv2.VideoCapture('video.mp4')
net, output_layers = load_yolo_model()
classes = load_classes()

# Запрос класса для распознавания
target_class_name = input("Введите название класса для распознавания (на английском): ").strip().lower()
if target_class_name not in classes:
    print("Указанный класс не найден в списке поддерживаемых классов.")
    exit()

target_class_id = classes.index(target_class_name)

# Переменная для хранения траектории движения объекта
trajectory = []

# Подключение к базе данных
conn = connect_db()

frame_number = 0
detected_objects = set()  # Множество для хранения идентификаторов обнаруженных объектов

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Детекция объектов
    detections = detect_objects(frame, net, output_layers)

    # Фильтрация детекций по целевому классу
    detections = [d for d in detections if d[5] == target_class_id]

    # Преобразование детекций для трекера
    detections_for_tracker = np.array([[x, y, x+w, y+h, score] for x, y, w, h, score, cls_id in detections])

    # Обновление трекера
    tracked_objects = tracker.update(detections_for_tracker)

    # Отрисовка треков и траектории движения
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # Вычисление центра объекта
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Добавление текущей позиции в траекторию
        trajectory.append((center_x, center_y))

        # Запись координат в базу данных
        insert_coordinates(conn, int(obj_id), frame_number, center_x, center_y)

        # Вывод предупреждения при обнаружении нового объекта
        if obj_id not in detected_objects:
            detected_objects.add(obj_id)
            show_alert()
            #print(f"Warning: {target_class_name} Detected with ID {int(obj_id)}")

    # Отрисовка траектории движения
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
