import os
import math
import time
import uuid
import json
import argparse
import pickle
from datetime import datetime

import numpy as np
import airsim
import cv2
from scipy.spatial.transform import Rotation as R
import utils.box_np_ops

# 1. 📂 Set save paths (NuScenes directory structure)

class NuScenesMaker:
    def __init__(self, 
                 base_dir = "/home/yeongyoo/03_Dataset/03_FineTune/",
                 pkl_file = "nuscenes_infos_train.pkl",
                 cam_names= ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                 settings_path = "~/Documents/AirSim/settings.json",
                 vehicle_name = "Drone",
                 lidar_name = "Lidar1",
                 docker_path = "/dataset/",
                 frequency = 10,
                 classes = ["Car", "Bench", "Birch"]
                 ):
        self.base_dir = base_dir
        self.cam_names = cam_names
        self.period = 1/frequency
        
        for name in cam_names:
            os.makedirs(os.path.join(base_dir, "samples", name), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "samples", "LIDAR_TOP"), exist_ok=True)
        
        self.pickle_path = os.path.join(base_dir, pkl_file)
        
        self.sample_dir = os.path.join(base_dir, "samples")
        self.docker_path = docker_path
        
        self.classes = classes

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.armDisarm(True, vehicle_name=vehicle_name)
        
        self.lidar_name = lidar_name
        self.vehicle_name = vehicle_name
        self.sensors_config = self._load_sensor_configs(settings_path, vehicle_name, lidar_name)
        
        self.prev_token = ""
    
    
    
    def _load_sensor_configs(self, settings_path, vehicle_name, lidar_name):
        """Function to automatically extract sensor offsets by reading the AirSim settings.json file"""
        # Convert ~ to absolute home directory path depending on OS
        settings_path = os.path.expanduser(settings_path)
        
        # Raise an error if the file doesn't exist to prevent silent failures
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"🚨 Cannot find {settings_path}! Please check the path.")

        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            
        # Dig into "SimpleFlight" drone settings following the custom JSON structure
        drone_settings = settings.get("Vehicles", {}).get(vehicle_name, {})
        cameras = drone_settings.get("Cameras", {})
        
        configs = {}
        
        # 1. 📸 Extract camera settings
        for cam_name, cam_info in cameras.items():
            configs[cam_name] = {
                "t": [cam_info.get("X", 0.0), cam_info.get("Y", 0.0), cam_info.get("Z", 0.0)],
                "r": [cam_info.get("Pitch", 0.0), cam_info.get("Roll", 0.0), cam_info.get("Yaw", 0.0)]
            }
            
        # 2. 🧊 Extract LiDAR settings (using 'Lidar1' key based on JSON config)
        lidar_info = drone_settings.get("Sensors", {}).get(self.lidar_name, {})
        print(lidar_info)
        if lidar_info:
            # 🌟 Rename to 'LIDAR_TOP' internally to match NuScenes format!
            configs["LIDAR_TOP"] = {
                "t": [lidar_info.get("X", 0.0), lidar_info.get("Y", 0.0), lidar_info.get("Z", 0.0)],
                "r": [lidar_info.get("Pitch", 0.0), lidar_info.get("Roll", 0.0), lidar_info.get("Yaw", 0.0)]
            }
            
        return configs
    
    
    
    def save_images(self, scene_id, timestamp):
        
        requests = [airsim.ImageRequest(name, airsim.ImageType.Scene, False, False) for name in self.cam_names]
        responses = self.client.simGetImages(requests)
        
        file_paths = []
        
        for i, response in enumerate(responses):
            
            filename = f"{scene_id}__{self.cam_names[i]}__{timestamp}.jpg"
            save_path = os.path.join(self.sample_dir, self.cam_names[i], filename)
            
            rel_path = os.path.join('samples', self.cam_names[i], filename)
            file_paths.append(rel_path)
            
            # binary data -> image
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            
            # save!
            cv2.imwrite(save_path, img_rgb)
            print(f"✅ Save complete: {save_path}")
        
        return file_paths
    
    
    
    def airsim_pose_to_matrix(self, pose):
        """Magic function to convert AirSim Pose (Position + Quaternion) into a 4x4 transformation matrix"""
        t = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
        q = [pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val]
        
        # Convert quaternion to 3x3 rotation matrix
        rotation_matrix = R.from_quat(q).as_matrix()
        
        # Assemble 4x4 Homogeneous Transformation Matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T
    
    
    
    def config_to_matrix(self, cfg):
        """Function to convert static configurations into a 4x4 transformation matrix"""
        t = cfg["t"]
        # Convert Euler angles (Pitch, Roll, Yaw) to rotation matrix
        rotation_matrix = R.from_euler('xyz', cfg["r"], degrees=True).as_matrix()
        
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T
    
    
    
    def save_lidar(self, scene_id, timestamp):
        lidar_data = self.client.getLidarData(self.lidar_name, 
                                              vehicle_name= self.vehicle_name)
        
        if len(lidar_data.point_cloud) < 3:
            print("⚠️ No LiDAR data found!")
            return
        
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape(-1, 3) # [N, 3] shape
        padding = np.zeros((points.shape[0], 2), dtype=np.float32)

        filename = f"{scene_id}__LIDAR_TOP__{timestamp}.pcd.bin"
        filepath = os.path.join(self.sample_dir, "LIDAR_TOP", filename)
        rel_path = os.path.join("samples", "LIDAR_TOP", filename)

        # 4. Smash it into a binary file!
        points_5d = np.hstack([points, padding])
        points_5d.tofile(filepath)
        print(f"📡 LiDAR save complete: {filepath} ({points.shape[0]} points)")
        
        return rel_path, points
    
    
    
    def get_info(self, scene_id, timestamp, cam_paths, lidar_path, lidar_points):
        current_token = uuid.uuid4().hex
        
        # 🌟 1. Perfectly recreate root structure (radars added!)
        dic = {
            "lidar_path": os.path.join(self.docker_path, lidar_path), # Note: lidar_path here must be the absolute path inside the Docker container!
            "token": current_token, 
            "sweeps": [], 
            "cams": {},
            "radars": {} # Allocate empty space so BEVFusion doesn't spit out Key errors
        }
        
        drone_pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        T_ego2global = self.airsim_pose_to_matrix(drone_pose)
        
        ego2global_r = R.from_matrix(T_ego2global[:3, :3]).as_quat() 
        dic["ego2global_translation"] = T_ego2global[:3, 3].tolist()
        dic["ego2global_rotation"] = [ego2global_r[3], ego2global_r[0], ego2global_r[1], ego2global_r[2]] 

        T_lidar2ego = self.config_to_matrix(self.sensors_config["LIDAR_TOP"])
        lidar2ego_r = R.from_matrix(T_lidar2ego[:3, :3]).as_quat()
        dic["lidar2ego_translation"] = T_lidar2ego[:3, 3].tolist()
        dic["lidar2ego_rotation"] = [lidar2ego_r[3], lidar2ego_r[0], lidar2ego_r[1], lidar2ego_r[2]]
        
        # 🌟 2. Perfectly recreate cams substructure & force float64
        for cam_name, cam_path in zip(self.cam_names, cam_paths):
            if hasattr(self, "sensors_config") and cam_name in self.sensors_config:
                T_cam2ego = self.config_to_matrix(self.sensors_config[cam_name])
                cam2ego_r = R.from_matrix(T_cam2ego[:3, :3]).as_quat()
                cam2ego_trans = T_cam2ego[:3, 3].tolist()
                cam2ego_rot = [cam2ego_r[3], cam2ego_r[0], cam2ego_r[1], cam2ego_r[2]] 
            else:
                cam2ego_trans = [0.0, 0.0, 0.0]
                cam2ego_rot = [1.0, 0.0, 0.0, 0.0] 
            
            cam_intrinsic = np.array([
                [800.0,   0.0, 800.0],
                [  0.0, 800.0, 450.0],
                [  0.0,   0.0,   1.0]
            ], dtype=np.float64) # Match info_invest.txt specifications

            T_ego2lidar = np.linalg.inv(T_lidar2ego)
            T_cam2ego_mat = np.eye(4)
            T_cam2ego_mat[:3, 3] = cam2ego_trans
            T_cam2ego_mat[:3, :3] = R.from_quat([cam2ego_rot[1], cam2ego_rot[2], cam2ego_rot[3], cam2ego_rot[0]]).as_matrix()
            T_cam2lidar = T_ego2lidar @ T_cam2ego_mat

            dic["cams"][cam_name] = {
                "data_path": os.path.join(self.docker_path, cam_path), # Note: 'Absolute path'
                "type": cam_name,
                "sample_data_token": uuid.uuid4().hex,
                "sensor2ego_translation": cam2ego_trans,
                "sensor2ego_rotation": cam2ego_rot,
                "ego2global_translation": dic["ego2global_translation"], 
                "ego2global_rotation": dic["ego2global_rotation"],       
                "timestamp": timestamp,
                "sensor2lidar_rotation": T_cam2lidar[:3, :3].astype(np.float64), 
                "sensor2lidar_translation": T_cam2lidar[:3, 3].astype(np.float64),
                "cam_intrinsic": cam_intrinsic
            }
        
        # 🌟 3. Timestamp and Token (add prev key!)
        dic["timestamp"] = timestamp
        dic["prev_token"] = self.prev_token
        dic["prev"] = self.prev_token if self.prev_token else "" 
        self.prev_token = current_token
        
        gt_boxes, gt_names, gt_velocity, num_lidar_pts = self.get_bbox(lidar_points)

        # 🌟 4. Perfectly match dataset label datatypes (dtype) (Core!)
        if len(gt_boxes) > 0:
            dic["gt_boxes"] = gt_boxes.astype(np.float64)
            dic["gt_names"] = gt_names.astype("<U32")
            dic["gt_velocity"] = gt_velocity.astype(np.float64)
            dic["num_lidar_pts"] = num_lidar_pts.astype(np.int64)
        else:
            dic["gt_boxes"] = np.zeros((0, 7), dtype=np.float64)
            dic["gt_names"] = np.array([], dtype="<U32")
            dic["gt_velocity"] = np.zeros((0, 2), dtype=np.float64)
            dic["num_lidar_pts"] = np.zeros(0, dtype=np.int64)
            
        dic["num_radar_pts"] = np.zeros(len(gt_boxes), dtype=np.int64) 
        dic["valid_flag"] = np.ones(len(gt_boxes), dtype=bool) 

        return dic
    
    
    
    def get_bbox(self, lidar_points):
        gt_boxes, gt_names, gt_velocity = [], [], []
        
        base_sizes = {
            "Car": [4.0, 2.0, 1.5],     # Car is 4m x 2m x 1.5m
            "Birch": [0.5, 0.5, 4.0],   # Tree is thin and tall!
            "Bench": [1.5, 0.6, 0.8]    # Bench is long horizontally!
        }

        # 🌟 1. Iterate through target classes: Scour all ["Car", "Birch", "Bench"]!
        for cls_name in self.classes:
            obj_names = self.client.simListSceneObjects(f"{cls_name}.*")
            
            base_l, base_w, base_h = base_sizes.get(cls_name, [1.0, 1.0, 1.0])

            for obj in obj_names:
                pose = self.client.simGetObjectPose(obj)
                if math.isnan(pose.position.x_val): continue # To prevent errors

                x, y, z = pose.position.x_val, pose.position.y_val, pose.position.z_val
                _, _, yaw = airsim.to_eularian_angles(pose.orientation) 

                scale = self.client.simGetObjectScale(obj)
                # 💡 Note: If actual sizes vary by class, handling branches here is more accurate!
                # Sticking with the temporary logic for now!
                l, w, h = scale.x_val * base_l, scale.y_val * base_w, scale.z_val * base_h

                gt_boxes.append([x, y, z, l, w, h, yaw])
                # NuScenes hates mixed case names! Unify to lowercase ('car', 'birch', etc.)
                gt_names.append(cls_name.lower()) 
                gt_velocity.append([0.0, 0.0])

        # --- 2. NuScenes format & 🌟 Special Outlier Filtering 🌟 ---
        if len(gt_boxes) > 0:
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            gt_names = np.array(gt_names)
            gt_velocity = np.array(gt_velocity, dtype=np.float32)

            if lidar_points is not None:
                point_indices = box_np_ops.points_in_rbbox(lidar_points, gt_boxes)
                num_lidar_pts = point_indices.sum(axis=0).astype(np.int32)
                
                # 🌟 Filtering magic: Only objects with at least 1 LiDAR point get True!
                valid_mask = num_lidar_pts > 0
                
                # Things out of sight are permanently expelled from the ledger here!
                gt_boxes = gt_boxes[valid_mask]
                gt_names = gt_names[valid_mask]
                gt_velocity = gt_velocity[valid_mask]
                num_lidar_pts = num_lidar_pts[valid_mask]
            else:
                num_lidar_pts = np.zeros(len(gt_boxes), dtype=np.int32)
                
        # 3. Defense code: Prepare for the case where 0 cars remain after filtering
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_names = np.array([])
            gt_velocity = np.zeros((0, 2), dtype=np.float32)
            num_lidar_pts = np.zeros(0, dtype=np.int32)
            
        return gt_boxes, gt_names, gt_velocity, num_lidar_pts
    
    
        
    def run(self):
        
        # -----------------------------------------------------
        # 1. 📂 Check existing PKL file and load ledger (Append mode)
        # -----------------------------------------------------
        if os.path.exists(self.pickle_path):
            print(f"📂 Found existing ledger ({self.pickle_path})! Defrosting and appending.")
            with open(self.pickle_path, "rb") as f:
                dataset_dict = pickle.load(f)
        else:
            print("📝 No existing ledger. Creating a new one with NuScenes v1.0-mini specs!")
            dataset_dict = {
                "metadata": {"version": "v1.0-mini"},
                "infos": []
            }

        try:
            print("🚁 Starting data collection! (Press Ctrl+C to safely save and exit)")
            while True:
                start_time = time.time()
                
                now = datetime.now()
                scene_id = now.strftime("n%Y-%m-%d-%H-%M-%S-0400")
                timestamp = int(now.timestamp() * 1e6)
                
                cam_paths = self.save_images(scene_id, timestamp)
                lidar_path, lidar_points = self.save_lidar(scene_id, timestamp)
                
                if lidar_path is not None:
                    info = self.get_info(scene_id, timestamp, cam_paths, lidar_path, lidar_points)
                    dataset_dict["infos"].append(info)
                    print(f"✅ Frame added! (Total data stacked in current ledger: {len(dataset_dict['infos'])})")
                
                elapsed = time.time() - start_time
                time.sleep(max(0, self.period - elapsed))
                
        except KeyboardInterrupt:
            
            print("\n🛑 Collection end signal detected! Baking collected data into PKL...")
            with open(self.pickle_path, "wb") as f:
                pickle.dump(dataset_dict, f)
            print(f"🎉 Save complete! (Final frame count: {len(dataset_dict['infos'])})")



def parse_args():
    parser = argparse.ArgumentParser(description="AirSim NuScenes Dataset Maker")
    
    parser.add_argument("--base-dir", type=str, default="/home/yeongyoo/03_Dataset/03_FineTune/", help="the basic path to save the dataset")
    parser.add_argument("--pkl-file", type=str, default="nuscenes_infos_train.pkl", help="pickle file name to save")
    
    # 리스트 형태는 nargs='+'를 사용하면 터미널에서 띄어쓰기로 여러 개를 받을 수 있어!
    parser.add_argument("--cam-names", nargs="+", type=str, 
                        default=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                        help="camera name list to use")
                        
    parser.add_argument("--settings-path", type=str, default="~/Documents/AirSim/settings.json", help="AirSim settings.json path to extract sensor configurations")
    parser.add_argument("--vehicle-name", type=str, default="Drone", help="AirSim car(or drone) name")
    parser.add_argument("--lidar-name", type=str, default="Lidar1", help="LiDar name")
    parser.add_argument("--docker-path", type=str, default="/dataset/", help="The internal dataset path inside the Docker Container")
    parser.add_argument("--frequency", type=int, default=10, help="Data accumulation frequency (Hz)")
    
    parser.add_argument("--classes", nargs="+", type=str, default=["Car", "Bench", "Birch"], help="Class names to include in the dataset (must match AirSim object names)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    maker = NuScenesMaker(**vars(args))
    maker.run()