import cv2 as cv
import numpy as np
import typing

class PencilSketch:
    def __init__(self, blur_sigma: int = 5, ksize: typing.Tuple[int, int] = (0,0), sharpen_value: int = None, kernel: np.ndarray = None) -> None:
        self.blur_sigma = blur_sigma
        self.ksize = ksize
        self.sharpen_value = sharpen_value
        self.kernel = np.array([[0,-1,0], [-1, sharpen_value, -1], [0,-1,0]]) if kernel == None else kernel
    
    def dodge(self, front: np.ndarray, back: np.ndarray) -> np.ndarray:
        result = back*255.0/(255.0-front)
        result[result>255] = 255
        result[back==255] = 255

        return result.astype('uint8')
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        if self.sharpen_value is not None and isinstance(self.sharpen_value, int):
            inverted = 255 - image
            return 255 - cv.filter2D(src=inverted, ddepth=-1, kernel=self.kernel)
        return image
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        grayscale = np.array(np.dot(frame[..., :3], [0.299, 0.587, 0.114]), dtype=np.uint8)
        grayscale = np.stack((grayscale, )* 3, axis = -1)

        inverted_image = 255 - grayscale
        blur_image = cv.GaussianBlur(inverted_image, ksize=self.ksize, sigmaX=self.blur_sigma)

        final_image = self.dodge(blur_image, grayscale)

        sharpened_image = self.sharpen(final_image)

        return sharpened_image