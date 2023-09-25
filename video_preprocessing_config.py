import cv2
import pyrealsense2 as rs

class Config:
    def __init__(self):
        self.input_width = 1000
        self.input_height = 1000
        self.output_width = 640
        self.output_height = 480
        self.merge_image_width = 1280
        self.merge_image_height = 480
        self.video_source = "/home/m1sha/WORK/VideoPreProcessing/cut.avi"
        self.codec = (".mp4")
        self.output_path = None
        self.interpolation_method = cv2.INTER_AREA
        self.stream_format = (rs.format.z16, rs.format.bgr8)
        self.fps = 30
        self.device = 0
        self.use_realsense = False
        self.StereoSGBM_or_StereoBM = 0
        self.color_mode = cv2.COLOR_BGR2RGB
        self.blur_threshold = 120
