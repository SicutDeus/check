import numpy as np
import open3d as o3d



class Reconstruction:
    def __init__(self, config, depth):
        self.config = config
        self.T_frame_to_model = o3d.core.Tensor(np.eye(4))
        depth = o3d.t.geometry.Image(depth).to(self.config.device)
        try:
            self.model = o3d.t.pipelines.slam.Model(self.config.voxel_size,
                                                    self.config.block_resolution,
                                                    self.config.block_count,
                                                    self.T_frame_to_model,
                                                    self.config.device)
            self.input_frame = o3d.t.pipelines.slam.Frame(depth.rows,
                                                          depth.columns,
                                                          self.config.intrinsic_tensor,
                                                          self.config.device)
            self.raycast_frame = o3d.t.pipelines.slam.Frame(depth.rows,
                                                            depth.columns,
                                                            self.config.intrinsic_tensor,
                                                            self.config.device)
        except Exception as e:
            pass
        self.image_index = 0
        print('initializing done')

    def visualize_and_save_pcd(self):
        pcd = self.model.extract_pointcloud(self.config.surface_weight_thr)
        pcd = pcd.to_legacy()

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()

        try:
            o3d.io.write_point_cloud(self.config.output_path, pcd)
        except Exception as e:
            pass


    def set_new_frames(self, rgb, depth):
        rgb = o3d.t.geometry.Image(rgb).to(self.config.device)
        depth = o3d.t.geometry.Image(depth).to(self.config.device)
        self.input_frame.set_data_from_image('depth', depth)
        self.input_frame.set_data_from_image('color', rgb)
        print(f'image_index: {self.image_index}')

    def add_frames_to_model(self):
        try:
            if self.image_index > 0:
                print(1)
                result = self.model.track_frame_to_model(self.input_frame, self.raycast_frame,
                                                    self.config.depth_scale,
                                                    self.config.depth_max,
                                                    self.config.odometry_distance_thr)
                print(2)
                self.T_frame_to_model = self.T_frame_to_model @ result.transformation
            print(3)
            self.model.update_frame_pose(self.image_index, self.T_frame_to_model)
            print(4)
            self.model.integrate(self.input_frame,
                                 self.config.depth_scale,
                                 self.config.depth_max,
                                 self.config.trunc_voxel_multiplier)
            print(5)
            self.model.synthesize_model_frame(self.raycast_frame, self.config.depth_scale,
                                         self.config.depth_min, self.config.depth_max,
                                         self.config.trunc_voxel_multiplier, False)
        except Exception as e:
            print(e)

    def launch(self, rgb, depth):
        try:
            self.set_new_frames(rgb, depth)
            self.add_frames_to_model()
        except Exception as e:
            print('exception')
            return False
        self.image_index += 1
        return True

