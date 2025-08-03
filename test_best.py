import albumentations as A
import torch
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
from ultralytics.models.yolo.detect import DetectionTrainer

from salt_pepper_custom import SaltAndPepperNoise  # твой кастомный класс

A.SaltAndPepperNoise = SaltAndPepperNoise  # регистрируем его в albumentations


# Кастомный тренер с аугментациями
class CustomTrainer(DetectionTrainer):

    def get_transforms(self):
        transformations = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.Blur(blur_limit=3, p=0.1),
            A.HueSaturationValue(p=0.5),
            A.RandomFog(p=0.2),
            A.ISONoise(p=0.3),
            A.ImageCompression(),
            A.RandomShadow(p=0.2),
            A.RandomSnow(p=0.1),
            SaltAndPepperNoise(),  # твоя аугментация
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        print("✅ Albumentations трансформы установлены!")
        return Albumentations(transformations)


torch.backends.cudnn.benchmark = True  # Ускорит вычисления на 10-15%
torch.set_float32_matmul_precision('high')  # Для Tensor Cores

if __name__ == "__main__":
    print("✅ CUDA доступен:", torch.cuda.is_available())
    print("✅ Устройство:", torch.cuda.get_device_name(0))

    # Создаем YAML файл датасета
    yaml_content = r"""path: D:\\ITMO\\hakathons\\arhipelag_2025\datasets\\human
train: images/train
val: images/val
test: images/test

names:
  0: person

names:
  0: person
"""

    with open("human.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("✅ human.yaml создан заново!")

    # Конфигурация тренировки
    overrides = {
        "model": "yolov8s.pt",  # здесь можно поставить best.pt
        "data": "human.yaml",
        "imgsz": 480,  # размер картинки лучше 640, это я на свой слабый ноут поставила для теста
        "epochs": 100,
        "device": "mps",  # cuda канешна надо
        "optimizer": "AdamW",
        "conf": 0.3,  # больше 0.4 брать не будем, можем потерять важные данные, 0.3-0.4 надо нам брать, кажется
        "lr0": 1e-4,
        "single_cls": True,
        "patience": 0,
        "verbose": True,
        "pretrained": True,
        "workers": 12,
        "augment": True,
        "mosaic": 1.0,
        "mixup": 0.2
    }

    # Запускаем тренировку
    trainer = CustomTrainer(overrides=overrides)
    trainer.train()

    # Валидируем модель
    model = YOLO("runs/detect/train/weights/best.pt")
    metrics = model.val()

    # Предсказание
    results = model(
        "/Users/elizabethkulakova/hackathon_arch25/yammy_pandas_archipelag25/coco128/images/train2017/000000000643.jpg")  # тут можно какой-нибудь свой путь вставить, но это чисто для визуализации в самом конце
    results[0].show()
