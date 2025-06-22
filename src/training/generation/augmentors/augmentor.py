import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from skimage.util import random_noise


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
        blur_val = random.randint(5, 15)  # blur value random
        aug_img = cv2.blur(image, (blur_val, blur_val))
        return aug_img

    def rotate(self, image):
        rows, cols = image.shape[:2]
        Cx, Cy = rows, cols
        rand_angle = random.randint(-10, 10)  # random angle range
        M = cv2.getRotationMatrix2D((Cy // 2, Cx // 2), rand_angle, 1)  # center angle scale
        aug_img_rotate = cv2.warpAffine(image, M, (cols, rows))  # apply rotation matrix such as previously explained
        return aug_img_rotate

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

    def augment_image(self, image, methods: list):
        augmentor = Augmenter()
        for i in methods:
            if i == 0:
                image = augmentor.bilateral_filter(image)
            elif i == 1:
                image = augmentor.rotate(image)
            elif i == 2:
                image = augmentor.noise(image)
            elif i == 3:
                image = augmentor.blur(image)
            elif i == 4:
                image = augmentor.erode(image)
            elif i == 5:
                image = augmentor.convolute(image)
            elif i == 6:
                image = augmentor.dialate(image)
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




