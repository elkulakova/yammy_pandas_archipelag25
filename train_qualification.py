from ultralytics import YOLO
import os
import torch
import cv2
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Albumentations
from torch.utils.data import DataLoader
import requests
from key import TOKEN, CHAT_ID
from salt_pepper_custom import SaltAndPepperNoise #импорт класса
import albumentations as A

A.SaltAndPepperNoise = SaltAndPepperNoise


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

augment: True

albumentations:
  train:
    - name: RandomBrightnessContrast
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
    - name: Blur
      blur_limit: 3
      p: 0.1
    - name: ShiftScaleRotate
      shift_limit: 0.05
      scale_limit: 0.1
      rotate_limit: 15
      p: 0.5
    - name: HueSaturationValue
      hue_shift_limit: 20
      sat_shift_limit: 30
      val_shift_limit: 20
      p: 0.5
    - name: Fog
      p: 0.2
    - name: ISONoise
      intensity: (0.1, 0.3)
      p: 0.3
    - name: JpegCompression
      quality_lower: 70
      p: 0.2
    - name: RandomShadow
      p: 0.2
    - name: RandomSnow
      p: 0.1
    - name: SaltAndPepperNoise
      p: 0.1
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
        mosaic=1,
        mixup=0.2
    )

    print(train_results.speed)

    # Валидация
    metrics = model.val()

    results = model("1_001002.JPG")
    results[0].show()

    # path = model.export(format="onnx")
    # это код для телеграм бота, который отправляет результаты 
    # обучения модели в виде графиков, лучшую модель, 
    # результаты обучения по поколения в формате csv
    # но настроен пока не до конца, надо менять циферки под каждое обучение

    TOKEN = TOKEN
    CHAT_ID = CHAT_ID


    def send_message(text):
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text})


    def send_file(file_path):
        url = f"https://api.telegram.org/bot{TOKEN}/sendDocument"
        with open(file_path, "rb") as f:
            requests.post(url, data={"chat_id": CHAT_ID}, files={"document": f})


    # менять циферки тутъ :)
    send_message("Код завершил работу! 71")
    send_file("runs\\detect\\train71\\args.yaml")
    send_file("runs\\detect\\train71\\weights\\best.pt")
    send_file('runs\\detect\\train71\\BoxF1_curve.png')
    send_file('runs\\detect\\train71\\BoxP_curve.png')
    send_file('runs\\detect\\train71\\BoxPR_curve.png')
    send_file('runs\\detect\\train71\\BoxR_curve.png')
    send_file('runs\\detect\\train71\\confusion_matrix.png')
    send_file('runs\\detect\\train71\\confusion_matrix_normalized.png')
    send_file('runs\\detect\\train71\\labels.jpg')
    send_file('runs\\detect\\train71\\results.csv')
    send_file('runs\\detect\\train71\\results.png')
    send_file('runs\\detect\\train71\\train_batch0.jpg')
    send_file('runs\\detect\\train71\\train_batch1.jpg')
    send_file('runs\\detect\\train71\\train_batch2.jpg')
    send_file('runs\\detect\\train71\\val_batch0_labels.jpg')
    send_file('runs/detect/train71/val_batch1_labels.jpg')
    send_file('runs/detect/train71/val_batch1_pred.jpg')
    send_file('runs/detect/train71/val_batch2_labels.jpg')
    send_file('runs/detect/train71/val_batch2_pred.jpg')
