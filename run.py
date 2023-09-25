import cv2
import open3d as o3d
import numpy as np

from reconstruction import Reconstruction
from video_preprocessing import VideoPreprocessing
from reconstruction_config import ReconstructionConfig


def setup_video_preprocessing():
    video_preprocessing = VideoPreprocessing()
    #video_preprocessing.set_config_realsense()
    video_preprocessing.set_video_capture()
    #video_preprocessing.set_video_writer()
    return video_preprocessing


def transform_frame_for_o3d(frame, is_depth):
    frame = cv2.resize(frame, (640, 480))
    if is_depth:
        return np.ascontiguousarray(np.uint16(frame) * 256)
    else:
        return np.ascontiguousarray(np.uint8(frame))
    #return frame
    # color_frame = color_frame.astype(np.float32)
    # depth_frame = depth_frame.astype(np.float32)
    #return o3d.t.geometry.Image(np.asarray(frame))


def setup_reconstruction(video_preprocessing):
    reconstruction_config = ReconstructionConfig()
    img = video_preprocessing.get_colorrr_frame()
    depth_frame = video_preprocessing.get_depthhh_frame(img)
    depth_frame_o3d = transform_frame_for_o3d(depth_frame, True)
    return Reconstruction(reconstruction_config, depth_frame_o3d)


def run():
    video_preprocessing = setup_video_preprocessing()
    reconstruction = setup_reconstruction(video_preprocessing)
    count = 0
    while True:
        color_frame = video_preprocessing.get_colorrr_frame()
        depth_frame = video_preprocessing.get_depthhh_frame(color_frame)
        color_frame, _ = video_preprocessing.cut_frame(color_frame)

        if color_frame is not None and depth_frame is not None:
            color_frame = transform_frame_for_o3d(color_frame, False)
            depth_frame = transform_frame_for_o3d(depth_frame, True)


            success = reconstruction.launch(color_frame, depth_frame)
            count += 1
        if count == 5:
            break

    # video_preprocessing.video_writer_release()
    reconstruction.visualize_and_save_pcd()

cap = cv2.VideoCapture('cut.avi')
i = 0
while True:
    ret, frame = cap.read()
    cv2.imwrite(f'test/{i}.jpg', frame)
    import time
    time.sleep(20)
    i += 1

#run()
