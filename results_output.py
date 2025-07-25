"""
предположительно предсказание будет выдавать массив из кортежей вот таких: (image_id, xc, yc, w, h), лэйбл автоматически 0
img_width, img_height надо будет брать напрямую из директорий, так как в предикте у нас все изображения должны быть одинакового размера
мб тогда сделать предварительную таблицу, в которую внесем img_id, w_img, h_img? тогда надо будет xc, yc, w, h как-то рассчитывать тоже с начальными параметрами (хотя пропорции не изменятся по идее, так что пока делаем следующее)
"""
import pandas as pd
import os
from PIL import Image
from typing import List, Tuple
from directories import args

def create_table(indir: str) -> pd.DataFrame:
    """
        Создает таблицу с метаданными изображений и пустыми полями для предсказаний.

        Args:
            indir (str): Путь к директории с изображениями

        Returns:
            pd.DataFrame: Таблица с колонками:
                image_id, xc, yc, w, h, label, score, time_spent, w_img, h_img
    """
    basic_data = []
    for img_file in os.listdir(indir):
        if img_file.lower().endswith('.jpeg'):
            full_path = os.path.join(indir, img_file)
            name = os.path.splitext(img_file)[0]
            try:
                with Image.open(full_path) as img:
                    width, height = img.size

                basic_data.append({'image_id': name, 'xc': None, 'yc': None, 'w': None, 'h': None, 'label': 0, 'score': None, 'time_spent': None, 'w_img': width, 'h_img': height})

            except Exception as e:
                print(f'error occurred while processing image {img_file}: {str(e)}')
                continue

    return pd.DataFrame(basic_data)

def results_table(preds: List[Tuple[str, float, float, float, float, float, float]]) -> pd.DataFrame:
    """
        Формирует таблицу результатов детекции в формате CSV.

        Параметры:
            preds: Список кортежей с предсказаниями (image_id, x_center, y_center, rect_width, rect_height, score, time_spent)

        Возвращает:
            DataFrame с колонками:
            image_id: уникальное имя файла изображения
            xc: центр ограничивающего прямоугольника по ширине изображения, разделенный на ширину изображения;
            yc: центр ограничивающего прямоугольника по высоте изображения, разделенный на высоту изображения;
            w: ширина ограничивающего прямоугольника, разделенная на ширину изображения;
            h: высота ограничивающего прямоугольника, разделенная на высоту изображения;
            label: код класса Объекта поиска;
            score: вероятность предсказания;
            time_spent: время обработки изображения, секунд;
            w_img: ширина изображения в пикселях;
            h_img: высота изображения в пикселях.
    """

    final_df = create_table(args.indir)
    pred_df = pd.DataFrame(preds, columns=['image_id', 'xc', 'yc', 'w', 'h', 'score', 'time_spent'])

    final_df = final_df.merge(pred_df, 'inner', 'image_id')

    return final_df

def save_result(df: pd.DataFrame) -> None:
    filepath = os.path.join(args.outdir, 'predictions.csv')
    df.to_csv(filepath, index=False)
