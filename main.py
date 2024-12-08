import cv2
from ultralytics import YOLO

# Завантаження моделі
model = YOLO('last_best_vehicle_model.pt')

# URL IP-камери
IP_CAMERA_URL = "YOUR_IP_CAMERA_URL"  # Замініть на ваш URL

# Словники для класів та кольорів
transport_classes = {
    0: 'car',
    1: 'truck',
    2: 'bus',
    3: 'motorcycle'
}

colors = {
    'car': (0, 255, 0),      
    'truck': (255, 0, 0),   
    'bus': (0, 0, 255),      
    'motorcycle': (255, 255, 0)  
}

def get_source():
    while True:
        print("\nВиберіть джерело відео:")
        print("1. Відеофайл")
        print("2. IP-камера")
        choice = input("Ваш вибір (1 або 2): ")
        
        if choice == "1":
            path = input("Введіть шлях до відеофайлу (або натисніть Enter для використання 222.mp4): ")
            return path if path else "222.mp4"
        elif choice == "2":
            return "http://192.168.0.2:8081"
        else:
            print("Невірний вибір. Спробуйте ще раз.")

def main():
    # Отримання джерела відео від користувача
    source_path = get_source()
    
    # Підключення до джерела відео
    print(f"\nПідключення до джерела: {source_path}")
    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print(f"Помилка: Не вдалося відкрити джерело відео: {source_path}")
        exit()

    print("\nДля виходу натисніть 'q' або 'Esc'")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Помилка: Не вдалося отримати кадр")
            break

        # Виконання детекції
        results = model(frame, conf=0.3)

        # Відображення результатів
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            class_id = int(class_id)
            
            if class_id in transport_classes:
                class_name = transport_classes[class_id]
                color = colors[class_name]
                label = f'{class_name}: {score:.2f}'
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Показ результату
        cv2.imshow('Vehicle Detection', frame)
        
        # Вихід при натисканні 'q' або 'Esc'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()