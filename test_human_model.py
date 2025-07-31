from ultralytics import YOLO
import os
import torch
import cv2
import albumentations as A
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Albumentations
from torch.utils.data import DataLoader
import requests


torch.backends.cudnn.benchmark = True  # Ускорит вычисления на 10-15%
torch.set_float32_matmul_precision('high')  # Для Tensor Cores



if __name__ == "__main__":

    print("✅ CUDA доступен:", torch.cuda.is_available())
    print("✅ Устройство:", torch.cuda.get_device_name(0))

    yaml_content = r"""path: D:\\ITMO\\hakathons\\arhipelag_2025\datasets\\human
train: images/train
val: images/val
test: images/test
names:
  0: person
"""

    # Создаем YAML
    with open("human.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("✅ human.yaml создан заново!")



    model = YOLO("yolov8n.pt")

    # Обучение
    train_results = model.train(
        data="human.yaml",  # Конфиг датасета
        epochs=50,           # Кол-во эпох
        imgsz=640,           # Размер изображений
        device='cuda',            # GPU 0
        batch=32,  
        workers=12,
        optimizer='AdamW',
        amp=True,
        single_cls=True,
        patience=0,
        deterministic=False,
        overlap_mask=False,
        pretrained=True,
        verbose=True,
        lr0=1e-4,


        augment=True,
        degrees=10,            # Поворот ±10°
        scale=0.5,             # Масштабирование (0.5–1.5)
        shear=2,               # Сдвиг по оси до 2°
        hsv_h=0.005,           # Сдвиг оттенка
        hsv_s=0.2,             # Сдвиг насыщенности
        hsv_v=0.2, 
        mosaic=1,
        mixup=0.2            # Сдвиг яркости
        
    )

    print(train_results.speed)

    # Валидация
    metrics = model.val()

    results = model("1_001002.JPG")
    results[0].show()

    # path = model.export(format="onnx")

    TOKEN = "8336957289:AAH6hShu1bDZwUd5KfMTPIs8FMJhyI2EwY4"
    CHAT_ID = "5179925114"


    def send_message(text):
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text})


    def send_file(file_path):
        url = f"https://api.telegram.org/bot{TOKEN}/sendDocument"
        with open(file_path, "rb") as f:
            requests.post(url, data={"chat_id": CHAT_ID}, files={"document": f})


    # пример использования в конце кода
    send_message("Код завершил работу! 66")
    send_file("runs\\detect\\train66\\args.yaml")
    send_file("runs\\detect\\train66\\weights\\best.pt")
    send_file('runs\\detect\\train66\\BoxF1_curve.png')
    send_file('runs\\detect\\train66\\BoxP_curve.png')
    send_file('runs\\detect\\train66\\BoxPR_curve.png')
    send_file('runs\\detect\\train66\\BoxR_curve.png')
    send_file('runs\\detect\\train66\\confusion_matrix.png')
    send_file('runs\\detect\\train66\\confusion_matrix_normalized.png')
    send_file('runs\\detect\\train66\\labels.jpg')
    send_file('runs\\detect\\train66\\results.csv')
    send_file('runs\\detect\\train66\\results.png')
    send_file('runs\\detect\\train66\\train_batch0.jpg')
    send_file('runs\\detect\\train66\\train_batch1.jpg')
    send_file('runs\\detect\\train66\\train_batch2.jpg')
    send_file('runs\\detect\\train66\\val_batch0_labels.jpg')
    send_file('runs/detect/train66/val_batch1_labels.jpg')
    send_file('runs/detect/train66/val_batch1_pred.jpg')
    send_file('runs/detect/train66/val_batch2_labels.jpg')
    send_file('runs/detect/train66/val_batch2_pred.jpg')
