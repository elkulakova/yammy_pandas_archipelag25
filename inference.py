from ultralytics import YOLO
from directories import args
import pandas as pd
import os
from typing import List

model = YOLO('best.pt')

results = model.predict(args.indir)

def names_array(indir: str) -> List[str]:
    """
        Создает таблицу с метаданными изображений и пустыми полями для предсказаний.

        Args:
            indir (str): Путь к директории с изображениями

        Returns:
            names Список названий фоток
    """
    names = []
    for img_file in os.listdir(indir):
        if img_file.lower().endswith('.jpeg'):
            name = os.path.splitext(img_file)[0]
            names.append(name)

    return names

def save_result(df: pd.DataFrame) -> None:
    filepath = os.path.join(args.outdir, 'predictions.csv')
    df.to_csv(filepath, index=False)

filenames = names_array(args.indir)
basic_data = []

for r, f in zip(results, filenames):
    h_im, w_im = r.orig_shape  # оригинальные размеры картинки

    time_spent = r.speed['total']

    if r.boxes:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:# class id
                score = round(float(box.conf[0]), 4)              # confidence
                x1, y1, x2, y2 = box.xyxy[0]           # координаты в формате xyxy
                x_center = ((x1 + x2) / 2) / w_im
                y_center = ((y1 + y2) / 2) / h_im
                width = (x2 - x1) / w_im
                height = (y2 - y1) / h_im

                basic_data.append(
                    {'image_id': f, 'xc': x_center, 'yc': y_center, 'w': width, 'h': height, 'label': 0, 'score': score,
                     'time_spent': time_spent, 'w_img': w_im, 'h_img': h_im})
    else:
        basic_data.append({'image_id': f, 'xc': None, 'yc': None, 'w': None, 'h': None, 'label': 0, 'score': None,
                           'time_spent': time_spent, 'w_img': w_im, 'h_img': h_im})  # пока непонятно, как Iou отреагирует на нан

final_df = pd.DataFrame(basic_data)

save_result(final_df)