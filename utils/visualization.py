import warnings
import colorsys
import random
import importlib
import cv2
import numpy as np


class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                message = """TensorboardX visualization is configured to use, but currently not installed on this machine. Please install the package by 'pip install tensorboardx' command or turn off the option in the 'config.json' file."""
                warnings.warn(message, UserWarning)
                logger.warn()
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data('{}/{}'.format(self.mode, tag), data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def draw_box(image, box, color, thickness=1, lineType=cv2.LINE_AA):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, lineType)


def draw_text(image, text, point, fg_color=(255, 255, 255), bg_color=(0, 0, 0), scale=0.4, thickness=1):
    x, y = int(point[0]), int(point[1])
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    right_bottom_point = (x + text_size[0], y + text_size[1])
    cv2.rectangle(image, (x, y), right_bottom_point, bg_color, cv2.FILLED)
    cv2.putText(image, text, (x, y + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, fg_color, thickness)


def draw_detections(image, detections, class_names=None, percent=False):
    '''
    display detections in image
    :param image:
    :param detections: N * [x1, y1, x2, y2, cls_index, cls_score]
    :param class_names:
    :param title:
    :return:
    '''
    # Number of boxes
    assert detections is not None
    N = detections.shape[0]

    # Generate random colors
    colors = random_colors(N)

    for i in range(N):
        color = colors[i]
        text_color = [1 - x for x in color]
        # bbox
        try:
            x1, y1, x2, y2, cls_index, cls_score = detections[i]
        except:
            x1, y1, x2, y2, cls_index = detections[i]
            cls_score = None
        if percent:
            x1 *= image.shape[1]
            y1 *= image.shape[0]
            x2 *= image.shape[1]
            y2 *= image.shape[0]
        draw_box(image, [x1, y1, x2, y2], color)

        # bbox's caption
        class_name = str(int(cls_index)) if class_names is None else class_names[int(cls_index)]
        if cls_score is None:
            caption = '%s' % class_name
        else:
            caption = '%s, %.3f' % (class_name, cls_score)
        draw_text(image, caption, (x1, y1), fg_color=text_color, bg_color=color)


if __name__ == "__main__":
    image = np.random.random([640, 480, 3])
    # colors = random_colors(10)
    # draw_box(image, [100, 50, 200, 300], colors[0])
    # draw_text(image, "test", (100, 50))
    detections = np.array([[100, 50, 200, 300, 0, 0.99], [200, 300, 300, 500, 1, 0.89]])
    draw_detections(image, detections, ["test1", "test2"])
    cv2.imshow("image", image)
    cv2.waitKey(-1)
