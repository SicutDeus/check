from video_preprocessing import VideoPreprocessing
import cv2

def run_test():
    test = VideoPreprocessing()

    img1 = cv2.imread("1.jpg")
    img2 = cv2.imread("2.jpg")
    img3 = cv2.imread("3.jpg")

    img1 = test.change_image_color_mode(img1, cv2.COLOR_BGR2GRAY)
    cv2.imshow('после изменений', img1)
    cv2.waitKey(0)
    img1 = test.change_image_color_mode(img1, cv2.COLOR_GRAY2BGR)

    test.is_image_blured(img1)
    test.is_image_blured(img3)

    img2 = test.set_image_size(img2)
    cv2.imshow("изображение с измененныым размером", img2)
    cv2.waitKey(0)

    merge_img = test.merge_images(cv2.imread("1.jpg"), cv2.imread("2.jpg"))
    cv2.imshow("склеенное изображение", merge_img)
    cv2.waitKey(0)

    img3 = test.image_histogram_alignment(img3)
    cv2.imshow("изображение с измененной контрастностью", img3)
    cv2.waitKey(0)

    test.set_video_capture()

    frame = test.get_colorrr_frame()
    depth_frame = test.get_depthhh_frame(frame)
    cv2.imshow("кадр глубины", depth_frame)
    cv2.waitKey(0)
