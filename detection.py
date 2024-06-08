import cv2

# Указать путь к исходному видеофайлу
input_video_path = '.mp4'
# Указать путь к выходному видеофайлу
output_video_path = 'out/.mp4'

# Открыть исходное видео
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Ошибка: не удалось открыть видеофайл.")
    exit()

# Получить ширину и высоту кадра видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создать объект VideoWriter для записи видео в формате .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

# Указать путь к видеофайлу

backSub = cv2.createBackgroundSubtractorMOG2()

# Открыть видеофайл
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Ошибка: не удалось открыть видеофайл.")
    exit()

while True:
    # Считать кадр
    ret, frame = cap.read()

    # Проверить, удалось ли считать кадр
    if not ret:
        print("Ошибка: не удалось считать кадр или видео закончилось.")
        break

    fg_mask = backSub.apply(frame)

    retval, mask_thresh = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY)

    # вычисление ядра
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Apply erosion
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    # Поиск контура
    contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100  # Define your minimum area threshold
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    # print(contours)
    #frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    frame_out = frame.copy()
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        frame_out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 3)

    # Отобразить кадр
    cv2.imshow('Видео', frame_out)

    out.write(frame_out)

    # ret, frame = cap.read()

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Освободить захват видео и закрыть все окна
cap.release()
cv2.destroyAllWindows()
