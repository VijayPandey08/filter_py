from pencilSketch import PencilSketch
from engine import Engine


if __name__ == "__main__":
    pencilSketch = PencilSketch(blur_sigma = 12, sharpen_value=5)
    selfieSegmentation = Engine(webcam_id=0, show = True, custom_objects = [pencilSketch])
    selfieSegmentation.run()