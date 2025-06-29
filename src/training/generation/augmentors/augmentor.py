import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from skimage.util import random_noise
import cv2

class Augmenter:

    def warp(self, image):
        rows, cols = image.shape[:2]
        tx = random.randint(-.25 * cols, .25 * cols)
        ty = random.randint(-.25 * rows, .25 * rows)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        aug_img_warp = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
        x, y = max(tx, 0), max(ty, 0)
        w, h = cols - abs(tx), rows - abs(ty)
        aug_img_warp = aug_img_warp[y:y + h, x:x + w]
        aug_img_warp = cv2.resize(aug_img_warp, (cols, rows))
        return aug_img_warp

    def noise(self, image):
        rows, cols = image.shape[:2]
        aug_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(aug_img)
        h += np.random.randint(0, 100, size=(rows, cols), dtype=np.uint8)
        s += np.random.randint(0, 20, size=(rows, cols), dtype=np.uint8)
        v += np.random.randint(0, 10, size=(rows, cols), dtype=np.uint8)
        aug_img = cv2.merge([h, s, v])
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_HSV2RGB)
        return aug_img

    def blur(self, image):
        blur_val = random.randint(1, 3)
        aug_img = cv2.blur(image, (blur_val, blur_val))
        return aug_img


    def erode(self, image):
        kernel = np.ones((1, 1), np.uint8)
        aug_img_erode = cv2.erode(image, kernel, iterations=1)
        return aug_img_erode

    def dialate(self, image):
        kernel = np.ones((1, 1), np.uint8)
        aug_img_dilate = cv2.dilate(image, kernel, iterations=1)
        return aug_img_dilate

    def convolute(self, image):
        kernel = np.ones((10, 10), np.float32) / 100
        aug_img_convolute = cv2.filter2D(image, -1, kernel)
        return aug_img_convolute

    def median_filter(self, image):
        noise_img = random_noise(image, mode='s&p', amount=0.3)
        noise_img = np.array(255 * noise_img, dtype='uint8')
        aug_img_median = cv2.medianBlur(noise_img, 5)
        return aug_img_median

    def bilateral_filter(self, image):
        return cv2.bilateralFilter(image, 20, 200, 300)

    def canny_edge(self, image):
        return cv2.Canny(image, 100, 200)

    def change_colors(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        h_shift = random.randint(-10, 10)
        s_shift = random.randint(-40, 40)
        v_shift = random.randint(-30, 30)

        h = (h.astype(np.int16) + h_shift)
        h = np.clip(h, 0, 179).astype(np.uint8)
        s = np.clip(s.astype(np.int16) + s_shift, 0, 255).astype(np.uint8)
        v = np.clip(v.astype(np.int16) + v_shift, 0, 255).astype(np.uint8)

        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def random_augment(self, image):
        methods = [
            self.bilateral_filter,
            self.change_colors,
            self.noise,
            self.blur,
            self.erode,
            self.convolute,
            self.dialate,
            self.median_filter
        ]
        num_augmentations = random.randint(1, 3)
        selected_methods = random.sample(methods, num_augmentations)

        random.shuffle(selected_methods)
        for method in selected_methods:
            image = method(image)

        return image

    def augment_images(self, n_of_images_output: int, image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        images = []
        for i in range(n_of_images_output):
            mutations = self.generate_mutations()
            img = self.augment_image(image, mutations)
            images.append(img)
        return images

    def generate_mutations(self) -> list:
        cur_mutations: list = []
        max_nr_of_mutations = random.randint(0, 4)
        while len(cur_mutations) != max_nr_of_mutations:
            mutation = random.randint(0, 6)
            if len(cur_mutations) < 2:
                mutation = random.randint(0, 5)
            if mutation not in cur_mutations:
                cur_mutations.append(mutation)
        return cur_mutations

    def format_image(self, image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        return image
