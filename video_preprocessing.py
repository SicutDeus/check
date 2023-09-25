import cv2
import os
import numpy as np
import pyrealsense2 as rs
from video_preprocessing_config import Config
class VideoPreprocessing:

    def __init__(self):
        self.cap = None
        self.writer = None
        self.config = Config()
        self.pipeline = rs.pipeline()
        self.iter = 0

    def set_config_realsense(self):
        depth_format, color_format = self.config.stream_format
        rs.config.enable_stream(rs.stream.depth, self.config.input_width, self.config.input_height,
                                depth_format, self.config.fps)
        rs.config.enable_stream(rs.stream.color, self.config.input_width, self.config.input_height,
                                color_format, self.config.fps)


    def change_image_color_mode(self, img, color_mode=None):
        try:
            color_mode = color_mode or self.config.color_mode
            return cv2.cvtColor(img, color_mode)
        except Exception as e:
            pass

    def is_image_blured(self, img):
        gray_img = self.change_image_color_mode(img, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray_img, cv2.CV_64F).var() < self.config.blur_threshold:
            return True
        return False

    def set_image_size(self, img, shape=None):
        return cv2.resize(
            img,
            dsize=shape or (self.config.input_width, self.config.input_height),
            interpolation=self.config.interpolation_method
        )

    def merge_images(self, img1, img2):
        try:
            stitcher = cv2.Stitcher.create()
            status, merged_image = stitcher.stitch([img1, img2])
            merged_image = self.set_image_size(merged_image, (self.config.merge_image_width,
                                                              self.config.merge_image_height))
            if status == cv2.Stitcher_OK:
                return merged_image
            return None
        except Exception as e:
            pass

    def image_histogram_alignment(self, img):
        contrast_img = cv2.equalizeHist(self.change_image_color_mode(img, cv2.COLOR_BGR2GRAY))
        contrast_img = self.change_image_color_mode(contrast_img, cv2.COLOR_GRAY2BGR)
        #contrast_img = np.hstack((img, contrast_img))
        return contrast_img

    def set_video_capture(self):
        try:
            if isinstance(self.config.video_source, str):
                self.cap = cv2.VideoCapture(self.config.video_source)
            else:
                if self.config.use_realsense:
                    self.set_config_realsense()
                    self.pipeline.start(rs.config())
                else:
                    self.cap = cv2.VideoCapture(self.config.video_source)
        except Exception as e:
            pass

    def set_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        try:
            self.writer = cv2.VideoWriter(
                self.config.output_path,
                fourcc,
                self.config.fps,
                (self.config.output_width, self.config.output_height)
            )
        except Exception as e:
            pass

    def write_video(self, img):
        try:
            if not self.writer:
                self.set_video_writer()
            self.writer.write(img)
        except Exception as e:
            pass

    def video_writer_release(self):
        self.videofile_count += 1
        filename, file_extension = self.config.output_path.split('.')
        self.config.output_path = f"{filename}_{self.videofile_count}.{file_extension}"
        self.writer.release()

    #нужон тест
    def set_video_capture_size(self, width=None, height=None):
        width = width or self.config.input_width
        height = height or self.config.input_height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_video_capture_fps(self):
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

    def get_depthhh_frame(self, img=None):
        try:
            if self.config.use_realsense:
                frame = self.pipeline.wait_for_frames()
                depth_frame = frame.get_depthhh_frame()
                return depth_frame
            else:
                left_img, right_img = self.cut_frame(img)
                if self.config.StereoSGBM_or_StereoBM == 0:
                    return self.StereoSGBM(left_img, right_img)
                else:
                    return self.StereoBM(left_img, right_img)
        except Exception as e:
            pass

    def get_colorrr_frame(self):
        if self.config.use_realsense:
            color_frame = self.pipeline.wait_for_frames()
            color_frame = color_frame.get_colorrr_frame()
            return np.asanyarray(color_frame.get_data())

        else:
            ret, color_frame = self.cap.read()
            cv2.imwrite(f'test/{self.iter}.jpg', color_frame)
            self.iter+=1
            if not ret:
                return None
            return color_frame

    def cut_frame(self, img):
        width = img.shape[1]
        half_width = width // 2
        left_image = img[:, :half_width]
        right_image = img[:, half_width:]
        return left_image, right_image

    def StereoSGBM(self, left_img, right_img):
        block_size = 3
        min_disp = -176
        max_disp = 176
        num_disp = max_disp - min_disp
        uniquenessRatio = 15
        speckleWindowSize = 16
        speckleRange = 2
        disp12MaxDiff = 2

        stereo = cv2.StereoSGBM_create(
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
        left_img = self.change_image_color_mode(left_img, cv2.COLOR_BGR2GRAY)
        right_img = self.change_image_color_mode(right_img, cv2.COLOR_BGR2GRAY)
        disparity_SGBM = stereo.compute(left_img, right_img)
        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=125,
                                       beta=0, norm_type=cv2.NORM_MINMAX)  # увеличиваем контраст
        disparity_SGBM = np.uint8(disparity_SGBM)
        return disparity_SGBM  # комплексированное изображение

    def StereoBM(self, left_img, right_img):
        # Вариант исполнения получения кадра глубины
        stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)

        left_img = self.change_image_color_mode(left_img, cv2.COLOR_BGR2GRAY)
        right_img = self.change_image_color_mode(right_img, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(left_img, right_img)
        # Normalize the values to a range from 0..255 for a grayscale image
        disparity = cv2.normalize(disparity, disparity, alpha=125,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
        return np.uint8(disparity)
