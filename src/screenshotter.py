from PIL import ImageGrab, ImageFilter, ImageStat, Image
import cv2
import time
import numpy as np
# from pyCAIR import user_input
# import os

# colors in bgr
BGR_WHITE = (255, 255, 255)
BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_BLUE = (255, 0, 0)
BGR_YELLOW = (0, 255, 255)

QUIT_KEY_REPR = 'q'
QUIT_KEY_CODE = ord(QUIT_KEY_REPR)

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (15, 30)
fontScale = 1
FONT_COLOR = BGR_WHITE
lineType = 2

face_rect_color = BGR_RED
eye_rect_color = BGR_GREEN
eye_false_positive_color = BGR_YELLOW
body_rect_color = BGR_BLUE

face_cascade = cv2.CascadeClassifier(
    './data/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(
    './data/haarcascade_eye.xml'
)


def screenshot(*args, **kwargs):
    return ImageGrab.grab(*args, **kwargs)


def save(screenshot_image, *args, **kwargs):
    screenshot_image.save(*args, **kwargs)


def grayscale_image(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)


def BGR_2_RGB(img_pixels):
    return cv2.cvtColor(img_pixels, cv2.COLOR_BGR2RGB)


def BGR_2_HSV(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV)


def BGR_2_GRAY(img_pixels):
    return cv2.cvtColor(img_pixels, cv2.COLOR_BGR2GRAY)


def canny_image(img, *args, **kwargs):
    return cv2.Canny(img, *args, **kwargs)


def laplacian_image(img):
    return cv2.Laplacian(img, cv2.CV_64F)


def get_color_tracking_mask(img, lower_color_range, upper_color_range):
    hsv = BGR_2_HSV(img)

    mask = cv2.inRange(hsv, lower_color_range, upper_color_range)
    return mask


def add_text(img, text, bottom_left=bottomLeftCornerOfText, font_color=FONT_COLOR, outline=None):
    if outline:
        outline_positions = [
            [-2, 0],
            [2, 0],
            [0, -2],
            [0, 2]
        ]
        for offset_x, offset_y in outline_positions:
            new_bottom_left = (
                bottom_left[0] + offset_x, bottom_left[1] + offset_y)
            cv2.putText(img, text, new_bottom_left, font,
                        fontScale, outline, lineType)
    cv2.putText(img, text, bottom_left,
                font, fontScale, font_color, lineType)
    return img


def get_rgb_image_pixels(bbox=None):
    screen_shot_pixels = np.array(screenshot(bbox=bbox))
    rgb = BGR_2_RGB(screen_shot_pixels)
    return rgb


def get_rgb_screen(bbox=None):
    rgb = get_rgb_image_pixels(bbox=bbox)
    return rgb


def get_edges_image_pixels(bbox=None, threshold_1=20, threshold_2=250, apertureSize=3, L2gradient=False):
    ss = screenshot(bbox=bbox)
    hsv = BGR_2_HSV(ss)
    canny = canny_image(hsv, threshold_1, threshold_2,
                        apertureSize=apertureSize, L2gradient=L2gradient)
    return canny


def get_edges(bbox=None):
    edges = get_edges_image_pixels(bbox=bbox, threshold_1=75, threshold_2=250)
    return edges


def draw_eyes(roi_color, eyes):
    for (eyeX, eyeY, eyeWidth, eyeHeight) in eyes:
        cv2.rectangle(roi_color, (eyeX, eyeY), (eyeX + eyeWidth,
                                                eyeY + eyeHeight), eye_rect_color, 2)


def draw_faces(original_image, grayscale, faces):
    for (x, y, w, h) in faces:
        original_image = cv2.rectangle(
            original_image, (x, y), (x + w, y + h), face_rect_color, 2)
        roi_gray = grayscale[y:y + h, x:x + w]
        roi_color = original_image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 1)
        draw_eyes(roi_color, eyes)
    return original_image


def process_faces(grayscale_img):
    return face_cascade.detectMultiScale(grayscale_img, 1.25, 2)


def process_and_draw_faces(img, gray):
    faces = process_faces(gray)
    return draw_faces(img, gray, faces)


def get_faces(bbox=None):
    ss = screenshot(bbox=bbox)
    screen_shot_pixels = np.array(ss)
    rgb = BGR_2_RGB(screen_shot_pixels)
    gray = grayscale_image(rgb)
    screen = process_and_draw_faces(rgb, gray)
    return screen


def get_color(lower_color_bound, upper_color_bound, bbox=None):
    ss = screenshot(bbox=bbox)
    screen_shot_pixels = np.array(ss)
    rgb = BGR_2_RGB(screen_shot_pixels)
    color_mask = get_color_tracking_mask(
        rgb, lower_color_bound, upper_color_bound)
    color_tracked_result = cv2.bitwise_and(rgb, rgb, mask=color_mask)
    return color_tracked_result


def get_blue(bbox=None):
    lower_blue = np.array([110, 10, 10])
    upper_blue = np.array([130, 255, 255])
    return get_color(lower_blue, upper_blue, bbox=bbox)


def get_red(bbox=None):
    lower_red = np.array([0, 10, 10])
    upper_red = np.array([20, 255, 255])
    return get_color(lower_red, upper_red, bbox=bbox)


def get_green(bbox=None):
    lower_green = np.array([25, 10, 10])
    upper_green = np.array([100, 255, 255])
    return get_color(lower_green, upper_green, bbox=bbox)


def get_rank(bbox=None, size=3, rank=0):
    ss = screenshot(bbox=bbox).filter(ImageFilter.RankFilter(size, rank))
    return BGR_2_RGB(np.array(ss))


def get_median(bbox=None, size=3):
    ss = screenshot(bbox=bbox).filter(ImageFilter.MedianFilter(size))
    return BGR_2_RGB(np.array(ss))


def add_stats(img_pixels):
    image_stats = ImageStat.Stat(Image.fromarray(img_pixels))
    image_stats_text = """
* Min/max values for each band in the image:
        {.extrema}
* Total number of pixels for each band in the image:
        {.count}
* Total number of pixels for each band in the image:
        {.sum}
* Squared sum of all pixels for each band in the image:
        {.sum2}
* Average (arithmetic mean) pixel level for each band in the image:
        {.mean}
* Median pixel level for each band in the image:
        {.median}
* RMS (root-mean-square) for each band in the image:
        {.rms}
* Variance for each band in the image:
        {.var}
* Standard deviation for each band in the image:
        {.stddev}
        """.format(*((image_stats, ) * 9))
    print(image_stats_text)
    index = 0
    for k, v in vars(image_stats).items():
        index += 1
        bottom_left = (
            bottomLeftCornerOfText[0], bottomLeftCornerOfText[1] + index * 40)
        add_text(img_pixels, '{}: {}'.format(k, v),
                 bottom_left=bottom_left, outline=(0, 0, 0))
    return img_pixels


def get_laplacian(bbox=None):
    ss = screenshot(bbox=bbox)
    rgb = BGR_2_RGB(np.array(ss))
    return laplacian_image(rgb)


def watch(bbox=None, mode=None, stats=None):
    def get_screen():
        if mode == 'rgb':
            return get_rgb_screen(bbox=bbox)
            # pass
            # return user_input(2, .5, 0, './capture.png', get_rgb_screen(bbox=bbox))
        if mode == 'edges':
            return get_edges(bbox=bbox)
        if mode == 'laplacian':
            return get_laplacian(bbox=bbox)
        if mode == 'faces':
            return get_faces(bbox=bbox)
        if mode == 'blue':
            return get_blue(bbox=bbox)
        if mode == 'red':
            return get_red(bbox=bbox)
        if mode == 'green':
            return get_green(bbox=bbox)
        if mode == 'rank':
            return get_rank(bbox=bbox)
        if mode == 'median':
            return get_median(bbox=bbox)
        return get_rgb_screen(bbox=bbox)

    print(f'Press {QUIT_KEY_REPR} to quit.')
    initial_time = last_time = time.time()
    counter = 0
    while (True):
        screen = get_screen()
        now = time.time()
        time_diff = now - last_time
        total_time = now - initial_time
        loops_per_second = counter / total_time if total_time != 0 else counter
        text = 'loop time: {:.4f}, LPS: {:.2f}'.format(
            time_diff, loops_per_second)
        last_time = now
        add_text(screen, text)
        counter += 1

        if stats:
            add_stats(screen)
        cv2.imshow('Watching {}'.format(mode), screen)

        if cv2.waitKey(25) & 0xFF == QUIT_KEY_CODE:
            cv2.destroyAllWindows()
            break

        # rgb = BGR_2_RGB(screen_shot)
        # screen = cv2.bitwise_and(rgb, rgb, mask=screen)
        # # screen = cv2.bitwise_not(rgb, rgb, mask=screen)
        # # lower_blue = np.array([110, 10, 10])
        # # upper_blue = np.array([130, 255, 200])
        # # color_mask = get_color_tracking_mask(rgb, lower_blue, upper_blue)
        # # color_tracked_result = cv2.bitwise_and(
        # #     rgb, rgb, mask=color_mask)
        # now = time.time()
        # time_this_loop = now - last_time
        # loops_per_second = counter / (now - initial_time)
        # add_text(screen, 'loop time: {:.4f}, LPS: {:.2f}'.format(
        #     time.time() - last_time, loops_per_second))
        # last_time = time.time()

        # # rgb = BGR_2_RGB(screen)
        # # cv2.imshow('window', rgb)

        # cv2.imshow('window', screen)
