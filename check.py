
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

skip_frame = 200
# Количество кадров для пропуска (костыль для работы с видео)
skip_sum = 0


# Счетчик пропущенных кадров
def main():
    VersionStereoClass = 1  # выбрать вариант реализации 1 или 0
    if VersionStereoClass == 1:
        StereoSGBM_func()
    else:
        StereoBM_func()


def StereoSGBM_func():
    video = cv.VideoCapture("cut.avi")
    # Вариант исполнения получения кадра глубины
    h = 0
    w = 0
    # Ширина и высота кадра
    x = 0
    x1 = 0
    y = 0
    # Отступы для первого кадра x y и для второго x1 y


    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    block_size = 3
    min_disp = -176
    max_disp = 176
    # Maximum disparity minus minimum disparity. The value is always greater than zero.
    # In the current implementation, this parameter must be divisible by 16.
    num_disp = max_disp - min_disp
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 15
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 16
    # Maximum disparity variation within each connected component.
    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    # Normally, 1 or 2 is good enough.
    speckleRange = 2
    disp12MaxDiff = 2
    skip_sum = 0

    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    # инициализации класса Стерео описание параметров с оф документации
    while (video.isOpened()):
        ret, frame = video.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        print("H:" + str(height) + "  W:" + str(width))
        skip_sum = skip_sum + 1 # счетчик скипнутых кадров
        if (skip_frame < skip_sum):
            h = int(height)
            w = int(width / 2) # делем на пополам потому что изображение сдвоенное
            x1 = w
            left_img = gray[y:y + h, x:x + w]   # получаем левое изображение из сдвоенного
            right_img = gray[y:y + h, x1:x1 + w]   # получаем правое изображение из сдвоенного
            disparity_SGBM = stereo.compute(left_img, right_img)
            # Normalize the values to a range from 0..255 for a grayscale image
            disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=125,
                                          beta=0, norm_type=cv.NORM_MINMAX)  # увеличиваем контраст
            disparity_SGBM = np.uint8(disparity_SGBM)
            res = disparity_SGBM # комплексированное изображение
            cv.imshow('left', res)
        else:
            pass
            #cv.imshow('original', gray)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()

def StereoBM_func():
    video = cv.VideoCapture("cut.avi")
    # Вариант исполнения получения кадра глубины
    h = 0
    w = 0
    # Ширина и высота кадра
    x = 0
    x1 = 0
    y = 0
    # Отступы для первого кадра x y и для второго x1 y
    skip_frame = 200
    # Количество кадров для пропуска (костыль для работы с видео)
    skip_sum = 0
    # Счетчик пропущенных кадров
    stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)

    while (video.isOpened()):
        ret, frame = video.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        print("H:" + str(height) + "  W:" + str(width))
        skip_sum = skip_sum + 1
        if (skip_frame < skip_sum):
            h = int(height)
            w = int(width / 2)
            x1 = w
            left_img = gray[y:y + h, x:x + w]
            right_img = gray[y:y + h, x1:x1 + w]
            disparity = stereo.compute(left_img, right_img)
            # Normalize the values to a range from 0..255 for a grayscale image
            disparity = cv.normalize(disparity, disparity, alpha=125,
                                          beta=0, norm_type=cv.NORM_MINMAX)
            disparity = np.uint8(disparity)
            res =  disparity
            cv.imshow('left', res)
        else:
            cv.imshow('original', gray)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()