import os
from argparse import *
from typing import List, Tuple

import pandas as pd
from ultralytics import YOLO


def input_directories() -> Tuple[str, str]:
    parser = ArgumentParser(description='paths to images and result')
    # пока что вариант для ситуации, когда в командную строку вводятся директории без ключей --input_dir/--output_dir
    parser.add_argument('indir', type=str, help='input dir with images')
    parser.add_argument('outdir', type=str, help='output dir to save the result')
    args = parser.parse_args()
    return args.indir, args.outdir


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


def save_result(df: pd.DataFrame, outdir: str) -> None:
    filepath = os.path.join(outdir, 'predictions.csv')
    df.to_csv(filepath, index=False)


def form_results(model: YOLO, indir: str) -> pd.DataFrame:
    results = model.predict(indir)
    filenames = names_array(indir)
    basic_data = []

    for r, f in zip(results, filenames):
        h_im, w_im = r.orig_shape  # оригинальные размеры картинки

        time_spent = r.speed['total']

        if r.boxes:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # class id
                    score = round(float(box.conf[0]), 4)  # confidence
                    x1, y1, x2, y2 = box.xyxy[0]  # координаты в формате xyxy
                    x_center = ((x1 + x2) / 2) / w_im
                    y_center = ((y1 + y2) / 2) / h_im
                    width = (x2 - x1) / w_im
                    height = (y2 - y1) / h_im

                    basic_data.append(
                        {'image_id': f, 'xc': x_center, 'yc': y_center, 'w': width, 'h': height, 'label': 0,
                         'score': score,
                         'time_spent': time_spent, 'w_img': w_im, 'h_img': h_im})
        else:
            basic_data.append({'image_id': f, 'xc': None, 'yc': None, 'w': None, 'h': None, 'label': 0, 'score': None,
                               'time_spent': time_spent, 'w_img': w_im,
                               'h_img': h_im})  # пока непонятно, как Iou отреагирует на нан
    return pd.DataFrame(basic_data)


if __name__ == "__main__":
    in_dir, out_dir = input_directories()
    our_model = YOLO('best.pt')
    final_df = form_results(our_model, in_dir)
    save_result(final_df, out_dir)
