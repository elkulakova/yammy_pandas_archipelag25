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
            # Улучшение контраста и адаптация к освещению
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),

            # Легкий шум и артефакты камеры/компрессии
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),
            A.ImageCompression(quality_range=(50, 95), p=0.3),
            A.Blur(blur_limit=3, p=0.1),
            SaltAndPepperNoise(),

            # Эмуляция реальных условий съёмки
            A.RandomShadow(p=0.2),
            A.RandomFog(alpha_coef=0.1, fog_coef_range=(0.1, 0.6), p=0.2),
            A.RandomRain(blur_value=2, drop_length=10, drop_width=1, p=0.15),
            A.RandomSnow(brightness_coeff=1.5, snow_point_range=(0.05, 0.15), p=0.1)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

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
"""

    with open("human.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("✅ human.yaml создан заново!")

    # Конфигурация тренировки
    overrides = {
        "model": "yolo8s.pt",  # здесь надо поставить best.pt
        "data": "human.yaml",
        "imgsz": 640,
        "epochs": 100,
        "batch": 32,
        "device": "cuda",
        "optimizer": "AdamW",
        "conf": 0.2,  # оказалось, что есть люди с 0.27...
        "lr0": 1e-4,
        "single_cls": True,
        "patience": 0,
        "verbose": True,
        "pretrained": True,
        "workers": 12,
        "augment": True,
        "mosaic": 0.5,
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
