import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2


class SaltAndPepperNoise(ImageOnlyTransform):
    def __init__(self):
        super().__init__()

    def apply(self, im: np.ndarray) -> np.ndarray:
        noisy_image = np.copy(im)

        # Determine the number of pixels to be affected by the noise (random amount between 0 and 30 percent)
        amount = np.random.uniform(0, 0.3)
        num_pixels = int(amount * im.shape[0] * im.shape[1])

        # Randomly select pixel locations to add noise to
        indices = np.random.choice(range(im.shape[0] * im.shape[1]), size=num_pixels, replace=False)

        # Split the image into its three color channels (BGR)
        b, g, r = cv2.split(noisy_image)

        # Add salt and pepper noise to the selected pixel locations in each color channel
        b[np.unravel_index(indices, im.shape[:2])] = np.random.choice([0, 255], size=(num_pixels,))
        g[np.unravel_index(indices, im.shape[:2])] = np.random.choice([0, 255], size=(num_pixels,))
        r[np.unravel_index(indices, im.shape[:2])] = np.random.choice([0, 255], size=(num_pixels,))

        # Merge the color channels back into a single image
        noisy_image = cv2.merge((b, g, r))

        return noisy_image

    def get_transform_init_args_names(self):
        return ()  # Пустой кортеж, так как нет параметров


"""Для использования кастомного класса перед обучением модели следует написать строки добавления этого класса в albumentations

from salt_pepper_custom import SaltAndPepperNoise #импорт класса
import albumentations as A
A.SaltAndPepperNoise = SaltAndPepperNoise #добавляем класс в уже существующие


Также в файле .yaml - добавить строку с классом в формате:
- name: SaltAndPepperNoise

Сам код для класса был взят с https://github.com/farbodYNSI/YOLO-augmenter.git"""