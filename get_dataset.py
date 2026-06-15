import os
import math
import time
import uuid
import json
import pickle
from datetime import datetime

import numpy as np
import airsim
import cv2
from scipy.spatial.transform import Rotation as R
import utils.box_np_ops

# 1. 📂 저장 경로 설정 (누씬즈 디렉토리 구조)

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
        """AirSim settings.json 파일을 읽어서 센서 오프셋을 자동 추출하는 함수"""
        # OS에 맞게 ~를 홈 디렉토리 절대 경로로 변환
        settings_path = os.path.expanduser(settings_path)
        
        # 파일이 없으면 에러 방지용으로 빈 딕셔너리 반환하거나 에러 뿜기
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"🚨 {settings_path} 파일이 없습니다! 경로를 확인해주세요.")

        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            
        # 캡틴의 JSON 구조를 따라 "SimpleFlight" 드론의 설정으로 파고들기
        drone_settings = settings.get("Vehicles", {}).get(vehicle_name, {})
        cameras = drone_settings.get("Cameras", {})
        
        configs = {}
        
        # 1. 📸 카메라 설정 긁어오기
        for cam_name, cam_info in cameras.items():
            configs[cam_name] = {
                "t": [cam_info.get("X", 0.0), cam_info.get("Y", 0.0), cam_info.get("Z", 0.0)],
                "r": [cam_info.get("Pitch", 0.0), cam_info.get("Roll", 0.0), cam_info.get("Yaw", 0.0)]
            }
            
        # 2. 🧊 라이다 설정 긁어오기 (캡틴의 JSON 기준 'Lidar1'이라는 키값 사용)
        lidar_info = drone_settings.get("Sensors", {}).get(self.lidar_name, {})
        print(lidar_info)
        if lidar_info:
            # 🌟 NuScenes 포맷에 맞춰 파이썬 내부에서는 'LIDAR_TOP'으로 이름 변경!
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
            print(f"✅ 저장 완료: {save_path}")
        
        return file_paths
    
    
    
    def airsim_pose_to_matrix(self, pose):
        """AirSim의 Pose(Position + Quaternion)를 4x4 변환 행렬로 바꾸는 마법의 함수"""
        t = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
        q = [pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val]
        
        # 쿼터니언을 3x3 회전 행렬로 변환
        rotation_matrix = R.from_quat(q).as_matrix()
        
        # 4x4 Homogeneous Transformation Matrix 조립
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T
    
    
    
    def config_to_matrix(self, cfg):
        """정적 설정을 4x4 변환 행렬로 바꾸는 함수"""
        t = cfg["t"]
        # 오일러 각도(Pitch, Roll, Yaw)를 회전 행렬로 변환
        rotation_matrix = R.from_euler('xyz', cfg["r"], degrees=True).as_matrix()
        
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T
    
    
    
    def save_lidar(self, scene_id, timestamp):
        lidar_data = self.client.getLidarData(self.lidar_name, 
                                              vehicle_name= self.vehicle_name)
        
        if len(lidar_data.point_cloud) < 3:
            print("⚠️ 라이다 데이터가 없습니다!")
            return
        
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape(-1, 3) # [N, 3] 형태
        padding = np.zeros((points.shape[0], 2), dtype=np.float32)

        filename = f"{scene_id}__LIDAR_TOP__{timestamp}.pcd.bin"
        filepath = os.path.join(self.sample_dir, "LIDAR_TOP", filename)
        rel_path = os.path.join("samples", "LIDAR_TOP", filename)

        # 4. 바이너리로 파일에 쾅! 박아버리기!
        points_5d = np.hstack([points, padding])
        points_5d.tofile(filepath)
        print(f"📡 라이다 저장 완료: {filepath} ({points.shape[0]} points)")
        
        return rel_path, points
    
    
    
    def get_info(self, scene_id, timestamp, cam_paths, lidar_path, lidar_points):
        current_token = uuid.uuid4().hex
        
        # 🌟 1. 루트 구조 완벽 재현 (radars 추가!)
        dic = {
            "lidar_path": os.path.join(self.docker_path, lidar_path), # 주의: 여기서 lidar_path는 Docker container 안에 들어갈 절대경로여야 함!
            "token": current_token, 
            "sweeps": [], 
            "cams": {},
            "radars": {} # BEVFusion이 Key 에러 뿜지 않도록 빈 공간 할당
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
        
        # 🌟 2. cams 하위 구조 완벽 재현 & float64 강제 적용
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
            ], dtype=np.float64) # info_invest.txt 규격 맞춤

            T_ego2lidar = np.linalg.inv(T_lidar2ego)
            T_cam2ego_mat = np.eye(4)
            T_cam2ego_mat[:3, 3] = cam2ego_trans
            T_cam2ego_mat[:3, :3] = R.from_quat([cam2ego_rot[1], cam2ego_rot[2], cam2ego_rot[3], cam2ego_rot[0]]).as_matrix()
            T_cam2lidar = T_ego2lidar @ T_cam2ego_mat

            dic["cams"][cam_name] = {
                "data_path": os.path.join(self.docker_path, cam_path), # 주의: '절대경로'
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
        
        # 🌟 3. 시간 및 토큰 (prev 키 추가!)
        dic["timestamp"] = timestamp
        dic["prev_token"] = self.prev_token
        dic["prev"] = self.prev_token if self.prev_token else "" 
        self.prev_token = current_token
        
        gt_boxes, gt_names, gt_velocity, num_lidar_pts = self.get_bbox(lidar_points)

        # 🌟 4. 데이터셋 라벨 자료형(dtype) 완벽 매칭 (핵심!)
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
            "Car": [4.0, 2.0, 1.5],     # 차는 4m x 2m x 1.5m
            "Birch": [0.5, 0.5, 4.0],   # 나무는 얇고 높게!
            "Bench": [1.5, 0.6, 0.8]    # 벤치는 가로로 길게!
        }

        # 🌟 1. KFP 신메뉴 순회: ["Car", "Birch", "Bench"] 싹 다 뒤지기!
        for cls_name in self.classes:
            obj_names = self.client.simListSceneObjects(f"{cls_name}.*")
            
            base_l, base_w, base_h = base_sizes.get(cls_name, [1.0, 1.0, 1.0])

            for obj in obj_names:
                pose = self.client.simGetObjectPose(obj)
                if math.isnan(pose.position.x_val): continue # 에러 방지용

                x, y, z = pose.position.x_val, pose.position.y_val, pose.position.z_val
                _, _, yaw = airsim.to_eularian_angles(pose.orientation) 

                scale = self.client.simGetObjectScale(obj)
                # 💡 주의: 클래스마다 실제 크기가 다르다면 여기서 분기 처리를 해주는 게 더 정확해!
                # 일단은 캡틴의 임시 로직을 그대로 사용!
                l, w, h = scale.x_val * base_l, scale.y_val * base_w, scale.z_val * base_h

                gt_boxes.append([x, y, z, l, w, h, yaw])
                # NuScenes는 대문자 섞인 이름을 안 좋아해! 소문자로 통일 ('car', 'birch' 등)
                gt_names.append(cls_name.lower()) 
                gt_velocity.append([0.0, 0.0])

        # --- 2. 누씬즈 포맷 및 🌟 KFP 특제 불량품 필터링 🌟 ---
        if len(gt_boxes) > 0:
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            gt_names = np.array(gt_names)
            gt_velocity = np.array(gt_velocity, dtype=np.float32)

            if lidar_points is not None:
                point_indices = box_np_ops.points_in_rbbox(lidar_points, gt_boxes)
                num_lidar_pts = point_indices.sum(axis=0).astype(np.int32)
                
                # 🌟 필터링 마법: 라이다 점이 1개라도 찍힌 객체만 True!
                valid_mask = num_lidar_pts > 0
                
                # 시야 밖에 있는 놈들은 여기서 장부에서 영구 제명됨!
                gt_boxes = gt_boxes[valid_mask]
                gt_names = gt_names[valid_mask]
                gt_velocity = gt_velocity[valid_mask]
                num_lidar_pts = num_lidar_pts[valid_mask]
            else:
                num_lidar_pts = np.zeros(len(gt_boxes), dtype=np.int32)
                
        # 3. 방어 코드: 필터링하고 났더니 차가 한 대도 안 남았을 경우를 대비
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_names = np.array([])
            gt_velocity = np.zeros((0, 2), dtype=np.float32)
            num_lidar_pts = np.zeros(0, dtype=np.int32)
            
        return gt_boxes, gt_names, gt_velocity, num_lidar_pts
    
    
        
    def run(self):
        
        # -----------------------------------------------------
        # 1. 📂 기존 PKL 파일 확인 및 장부 불러오기 (이어쓰기 모드)
        # -----------------------------------------------------
        if os.path.exists(self.pickle_path):
            print(f"📂 기존 장부({self.pickle_path})를 발견했습니다! 해동해서 이어서 작성합니다.")
            with open(self.pickle_path, "rb") as f:
                dataset_dict = pickle.load(f)
        else:
            print("📝 기존 장부가 없습니다. 누씬즈 v1.0-mini 규격으로 새 장부를 만듭니다!")
            dataset_dict = {
                "metadata": {"version": "v1.0-mini"},
                "infos": []
            }

        try:
            print("🚁 데이터 수집 시작! (Ctrl+C를 누르면 안전하게 저장하고 종료됩니다)")
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
                    print(f"✅ 프레임 추가 완료! (현재 장부에 쌓인 총 데이터: {len(dataset_dict['infos'])}개)")
                
                elapsed = time.time() - start_time
                time.sleep(max(0, self.period - elapsed))
                
        except KeyboardInterrupt:
            
            print("\n🛑 수집 종료 신호 감지! 지금까지 모은 데이터를 PKL로 굽습니다...")
            with open(self.pickle_path, "wb") as f:
                pickle.dump(dataset_dict, f)
            print(f"🎉 저장 완료! (최종 데이터 수: {len(dataset_dict['infos'])} 프레임)")



if __name__ == "__main__":
    maker = NuScenesMaker()
    maker.run()
