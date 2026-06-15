import argparse
import copy
import os
import json
import math
from datetime import datetime
import time
import collections

import cv2
import mmcv
import numpy as np
import torch
import airsim
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes


class BEVFusionEvaluator:
    def __init__(self, args, opts):
        """초기화 및 환경 설정"""
        dist.init()
        self.args = args
        
        # 설정 로드 및 병합
        configs.load(args.config, recursive=True)
        configs.update(opts)
        self.cfg = Config(self._recursive_eval(configs), filename=args.config)

        # GPU 설정
        torch.backends.cudnn.benchmark = self.cfg.cudnn_benchmark
        torch.cuda.set_device(dist.local_rank())

        # 데이터 로더 및 모델 초기화
        self.model = self._build_and_load_model()
        print(self.model)
        
        # AirSimulator 설정
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.armDisarm(True, vehicle_name= args.vehicle_name)
        self.sensors_config = self._load_sensor_configs(args.settings_path, args.vehicle_name, args.lidar_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # -------------- SIMULATOR RELATED -------------------
    
    def _load_sensor_configs(self, settings_path, vehicle_name, lidar_name):
        settings_path = os.path.expanduser(settings_path)
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"🚨 {settings_path} 파일이 없습니다!")

        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            
        drone_settings = settings.get("Vehicles", {}).get(vehicle_name, {})
        cameras = drone_settings.get("Cameras", {})
        
        configs = {}
        
        # 📸 1. 카메라 설정 긁어오기 (🌟 Y, Z 좌표와 Yaw, Roll 부호 반전!)
        for cam_name, cam_info in cameras.items():
            configs[cam_name] = {
                "t": [cam_info.get("X", 0.0), -cam_info.get("Y", 0.0), -cam_info.get("Z", 0.0)],
                "r": [cam_info.get("Pitch", 0.0), -cam_info.get("Roll", 0.0), -cam_info.get("Yaw", 0.0)]
            }
            
        # 🧊 2. 라이다 설정 긁어오기 (🌟 여기도 부호 반전!)
        lidar_info = drone_settings.get("Sensors", {}).get(self.args.lidar_name, {})
        if lidar_info:
            configs["LIDAR_TOP"] = {
                "t": [lidar_info.get("X", 0.0), -lidar_info.get("Y", 0.0), -lidar_info.get("Z", 0.0)],
                "r": [lidar_info.get("Pitch", 0.0), -lidar_info.get("Roll", 0.0), -lidar_info.get("Yaw", 0.0)]
            }
            
        return configs
    
    @property
    def images(self):
        requests = [airsim.ImageRequest(name, airsim.ImageType.Scene, False, False) for name in self.args.cam_names]
        responses = self.client.simGetImages(requests)
        
        images = []
        for response in responses:
            # 1. 반환된 1차원 바이트 데이터를 8비트 정수형 numpy 배열로 변환
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            
            # 2. 이미지의 원래 해상도(높이, 너비)와 3채널(RGB)에 맞게 형태(Shape) 재조정
            img_rgb = img1d.reshape(response.height, response.width, 3)
            
            # 3. 리스트에 차곡차곡 추가
            images.append(img_rgb)
            
        return images
    
    @property
    def lidar(self):
        accumulated_points = []
        last_timestamp = 0  # 🌟 방금 찍은 사진의 시간을 기억할 변수
        frames_collected = 0
        
        # 무한 루프에 빠지지 않도록 최대 시도 횟수 제한 (보험용)
        max_retries = 50 
        retries = 0
        
        # 🌟 정확히 '서로 다른 시간'의 데이터를 10장 모을 때까지 반복!
        while frames_collected < 10 and retries < max_retries:
            lidar_data = self.client.getLidarData(self.args.lidar_name, vehicle_name=self.args.vehicle_name)
            
            # 🌟 [핵심] 데이터가 있고, 방금 찍은 사진과 '시간(timestamp)'이 다를 때만 수집!
            if len(lidar_data.point_cloud) >= 3 and lidar_data.time_stamp != last_timestamp:
                points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
                
                # 물리치료 (좌표축 뒤집기)
                points[:, 1] = -points[:, 1]
                points[:, 2] = -points[:, 2]
                
                accumulated_points.append(points)
                
                # 새로운 사진의 시간을 기억해두기!
                last_timestamp = lidar_data.time_stamp
                frames_collected += 1  # 성공적으로 한 장 수집 완료!
            
            # 엔진이 틱을 넘길 수 있도록 아주 짧게 대기 (0.005초)
            time.sleep(0.005)
            retries += 1
            
        if not accumulated_points:
            return None
            
        # 🌟 서로 다른 10개의 찰나의 순간들이 완벽하게 합쳐짐!
        merged_points = np.vstack(accumulated_points)
        
        # BEVFusion 모델 양식에 맞게 패딩 덧붙이기
        padding = np.zeros((merged_points.shape[0], 2), dtype=np.float32)
        return np.hstack([merged_points, padding])
    
    
    def _get_transformation_matrix(self, t, r):
        """
        [X, Y, Z]와 [Pitch, Roll, Yaw](Degree)를 받아 4x4 변환 행렬(Tensor)을 반환합니다.
        """
        # 1. Degree를 Radian으로 변환
        pitch, roll, yaw = map(math.radians, r)
        
        # 2. 회전 행렬 (Rotation Matrix) 계산 (일반적인 Z-Y-X 오일러 각 기준)
        # AirSim 환경에 맞춰 Roll(X), Pitch(Y), Yaw(Z) 축 회전
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(roll), -math.sin(roll)],
                        [0, math.sin(roll), math.cos(roll)]])
                        
        R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]])
                        
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                        [math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1]])
                        
        R = R_z @ R_y @ R_x
        
        # 3. 4x4 행렬 (Transformation Matrix) 조립
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R        # 좌상단 3x3은 회전
        T[:3, 3] = t         # 우상단 3x1은 이동(Translation)
        
        return torch.tensor(T, dtype=torch.float32, device=self.device)
    
    
    def _get_camera_intrinsics(self, cam_name, img_width=704, img_height=256, orig_width=800, orig_height=450):
        cam_info = self.client.simGetCameraInfo(cam_name, vehicle_name=self.args.vehicle_name)
        fov_rad = math.radians(cam_info.fov)
        
        # 1. 원래 해상도(800x450) 기준의 초점 거리
        fx_orig = orig_width / (2.0 * math.tan(fov_rad / 2.0))
        fy_orig = fx_orig 
        
        # 2. Resize 비율만큼 강제로 스케일링!
        fx = fx_orig * (img_width / orig_width)
        fy = fy_orig * (img_height / orig_height)
        
        cx = img_width / 2.0
        cy = img_height / 2.0
        
        K = torch.eye(4, dtype=torch.float32, device=self.device)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        
        return K
    
    
    def get_batch_gpu(self, current_images, current_lidar):
        from torchvision import transforms
        
        batch_gpu = {}
        N_cams = len(self.args.cam_names) # 6대의 카메라
        
        # getting transformed images
        transform = transforms.Compose([
            transforms.ToTensor(), # HWC -> CHW 변환 및 0~1 스케일링
            transforms.Resize((256, 704)), # 🌟 BEVFusion 핵심 요구 해상도!
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensors = [transform(img) for img in current_images]
        batch_gpu['img'] = torch.stack(img_tensors, dim=0).unsqueeze(0).to(self.device)
        
        # getting transformed points
        batch_gpu['points'] = [torch.tensor(current_lidar, dtype=torch.float32).to(self.device)]
        
        
        # getting lidar to ego
        lidar_cfg = self.sensors_config["LIDAR_TOP"]
        lidar2ego = self._get_transformation_matrix(lidar_cfg["t"], lidar_cfg["r"])
        batch_gpu['lidar2ego'] = lidar2ego.unsqueeze(0) # [1, 4, 4] 형태로 저장
        
        
        # getting camera to ego, ego to camera, lidar to camera
        camera2ego_list = []
        lidar2camera_list = []
        camera2lidar_list = []
        
        body2optical = torch.tensor([
            [0, -1,  0,  0], # Optical X(오른쪽) = -Body Y(왼쪽)
            [0,  0, -1,  0], # Optical Y(아래) = -Body Z(위)
            [1,  0,  0,  0], # Optical Z(앞) = Body X(앞)
            [0,  0,  0,  1]
        ], dtype=torch.float32, device=self.device)

        for cam_name in self.args.cam_names:
            cam_cfg = self.sensors_config[cam_name]
            cam2ego = self._get_transformation_matrix(cam_cfg["t"], cam_cfg["r"])
            camera2ego_list.append(cam2ego)
            
            ego2cam = torch.linalg.inv(cam2ego) 
            l2c_body = torch.matmul(ego2cam, lidar2ego) # 여기까지는 Body 기준!
            
            # 🌟 여기서 Body 프레임을 Optical 프레임으로 꺾어줍니다!
            l2c_optical = torch.matmul(body2optical, l2c_body)
            lidar2camera_list.append(l2c_optical)
            
            c2l = torch.linalg.inv(l2c_optical)
            camera2lidar_list.append(c2l)

        # stacking a list into the form of GPU tensor [1, 6, 4, 4] 
        batch_gpu['camera2ego'] = torch.stack(camera2ego_list, dim=0).unsqueeze(0)
        batch_gpu['lidar2camera'] = torch.stack(lidar2camera_list, dim=0).unsqueeze(0)
        batch_gpu['camera2lidar'] = torch.stack(camera2lidar_list, dim=0).unsqueeze(0)
        
        # Camera Intrinsics 
        intrinsics_list = []
        for cam_name in self.args.cam_names:
            # 주의: BEVFusion 리사이즈 해상도인 704x256에 맞춰서 계산해야 모델이 찰떡같이 알아먹음!
            K = self._get_camera_intrinsics(cam_name, img_width=704, img_height=256)
            intrinsics_list.append(K)
            
        # [6, 4, 4] 형태로 쌓고, 배치 차원 [1, 6, 4, 4]로 늘려줌
        batch_gpu['camera_intrinsics'] = torch.stack(intrinsics_list, dim=0).unsqueeze(0)
        
        #  (Intrinsics @ Lidar2Camera)
        batch_gpu['lidar2image'] = torch.matmul(batch_gpu['camera_intrinsics'], batch_gpu['lidar2camera'])
        
        
        # 🔄 4. Augmentation 행렬 (추론 시엔 이미지/라이다 회전이 없으므로 단위행렬)
        eye_4x4 = torch.eye(4, dtype=torch.float32, device=self.device)
        batch_gpu['img_aug_matrix'] = eye_4x4.view(1, 1, 4, 4).repeat(1, N_cams, 1, 1)
        batch_gpu['lidar_aug_matrix'] = eye_4x4.view(1, 4, 4)
        
        
        # 📝 5. 메타데이터 (모델 내부에서 timestamp나 토큰을 참조할 때 에러 방지용)
        batch_gpu['metas'] = [{
            'timestamp': datetime.now().timestamp(),
            'token': 'airsim_birthday_token',
            'box_mode_3d': 0, # MMDetection3D의 LiDAR 박스 모드
            'box_type_3d': LiDARInstance3DBoxes
        }]
        
        batch_gpu['depths'] = torch.zeros(1, device='cuda')
        
        return batch_gpu
    
    
    # -------------------------- MODEL RELATED ---------------------------
    
    def _recursive_eval(self, obj, globals_dict=None):
        """재귀적으로 설정을 평가하여 파싱"""
        if globals_dict is None:
            globals_dict = copy.deepcopy(obj)

        if isinstance(obj, dict):
            for key in obj:
                obj[key] = self._recursive_eval(obj[key], globals_dict)
        elif isinstance(obj, list):
            for k, val in enumerate(obj):
                obj[k] = self._recursive_eval(val, globals_dict)
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            obj = eval(obj[2:-1], globals_dict)
            obj = self._recursive_eval(obj, globals_dict)
        return obj
    
    def _build_and_load_model(self):
        """모델 빌드 및 가중치 로드"""
        model = build_model(self.cfg.model)
        load_checkpoint(model, self.args.checkpoint, map_location="cpu")
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()
        return model

    def _extract_predictions(self, outputs):
        """Prediction 데이터 추출"""
        bboxes, labels, masks = None, None, None

        if "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if self.args.bbox_classes is not None:
                indices = np.isin(labels, self.args.bbox_classes)
                bboxes, scores, labels = bboxes[indices], scores[indices], labels[indices]

            if self.args.bbox_score is not None:
                indices = scores >= self.args.bbox_score
                bboxes, scores, labels = bboxes[indices], scores[indices], labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

        if "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy() >= self.args.map_score

        return bboxes, labels, masks
    
    def _visualize_realtime(self, data, metas, name, bboxes, labels, masks, raw_images):
        """AirSim 실시간 데이터를 위한 메모리 직접 시각화 함수"""
        
        # 📸 1. 카메라 시각화 (저장 경로 자동 생성 기능 포함)
        for k, image in enumerate(raw_images):
            out_path = os.path.join(self.args.out_dir, f"camera-{k}", f"{name}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # 🌟 [수정된 부분] 도화지를 모델 투영 행렬 기준(704x256)으로 맞춰주기!
            image_resized = cv2.resize(image, (704, 256))
            
            visualize_camera(
                out_path,
                image_resized, # 원본 image 대신 image_resized 투입!
                bboxes=bboxes,
                labels=labels,
                transform=metas["lidar2image"][k],
                classes=self.cfg.object_classes,
            )

        # 🧊 2. 라이다 시각화
        # DataContainer 껍데기가 없으므로 data["points"][0]에서 바로 뽑아냄
        lidar = data["points"][0].cpu().numpy() 
        out_path = os.path.join(self.args.out_dir, "lidar", f"{name}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        visualize_lidar(
            out_path,
            lidar,
            bboxes=bboxes,
            labels=labels,
            xlim=[self.cfg.point_cloud_range[d] for d in [0, 3]],
            ylim=[self.cfg.point_cloud_range[d] for d in [1, 4]],
            classes=self.cfg.object_classes,
        )

        # 🗺️ 3. BEV Map 시각화
        if masks is not None:
            out_path = os.path.join(self.args.out_dir, "map", f"{name}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            visualize_map(
                out_path,
                masks,
                classes=self.cfg.map_classes,
            )

    def run(self):
        """1초마다 실시간으로 AirSim 데이터를 가져와 추론 및 시각화하는 파이프라인"""
        print("🚀 실시간 AirSim BEVFusion 추론을 시작합니다! (종료하려면 터미널에서 Ctrl+C)")

        try:
            while True:
                loop_start = time.time()

                # 1. 📸🧊 현재 프레임 캡처 (API 중복 호출 방지를 위한 캐싱)
                current_images = self.images
                current_lidar = self.lidar

                if current_lidar is None: # 라이다가 덜 로딩됐으면 스킵
                    print("⚠️ 라이다 응답 대기 중...")
                    time.sleep(0.1)
                    continue

                # 2. 📦 딥러닝 모델용 텐서 배치 조립
                data = self.get_batch_gpu(current_images, current_lidar)
                
                # 메타데이터 추출 및 시각화용 투영 행렬(lidar2image) 복사
                metas = data["metas"][0]
                metas["lidar2image"] = data["lidar2image"][0].cpu().numpy()
                name = f"{metas['timestamp']}-{metas['token']}"
                
                print(f"📡 [AirSim] 프레임 처리 중: {name}")
                
                if 'points' in data and len(data['points']) > 0:
                    print("LiDAR Shape:", data['points'][0].shape)
                    if data['points'][0].shape[0] > 0:
                        # X, Y, Z 좌표의 최솟값, 최댓값을 확인하여 범위를 점검
                        print("X min/max:", data['points'][0][:, 0].min().item(), "/", data['points'][0][:, 0].max().item())
                        print("Y min/max:", data['points'][0][:, 1].min().item(), "/", data['points'][0][:, 1].max().item())
                        print("Z min/max:", data['points'][0][:, 2].min().item(), "/", data['points'][0][:, 2].max().item())
                    else:
                        print("포인트 클라우드 개수가 0입니다!")

                # 3. 🧠 BEVFusion 추론! (실시간은 GT가 없으므로 무조건 pred 모드)
                with torch.inference_mode():
                    outputs = self.model(**data)
                bboxes, labels, masks = self._extract_predictions(outputs)
                
                if bboxes is not None and len(bboxes) > 0:
                    print(f"\n📦 [Frame: {name}] 검출된 객체 수: {len(bboxes)}")
                    
                    # LiDARInstance3DBoxes 내부의 텐서를 numpy 배열로 변환
                    # 형태: [N, 9] -> (x, y, z, x_size, y_size, z_size, yaw, vx, vy)
                    box_data = bboxes.tensor.cpu().numpy()
                    
                    for i in range(len(bboxes)):
                        # 위치 (Center X, Y, Z)
                        x, y, z = box_data[i][0:3]
                        # 크기 (Length, Width, Height)
                        l, w, h = box_data[i][3:6]
                        # 회전 각도 (라디안)
                        yaw = box_data[i][6]
                        
                        # 클래스 이름 매칭 (예: Car, Pedestrian 등)
                        class_name = self.cfg.object_classes[labels[i]] if labels is not None else "Unknown"
                        
                        # 깔끔하게 줄을 맞춰서 출력!
                        print(f"  👉 [{i+1}] {class_name:^10} | "
                              f"위치(x,y,z): ({x:>6.2f}, {y:>6.2f}, {z:>6.2f}) | "
                              f"크기(l,w,h): ({l:>5.2f}, {w:>5.2f}, {h:>5.2f}) | "
                              f"각도: {yaw:>5.2f} rad")
                else:
                    print(f"\n📦 [Frame: {name}] 검출된 객체가 없습니다.")

                # 4. 🎨 시각화 및 이미지 저장
                self._visualize_realtime(data, metas, name, bboxes, labels, masks, current_images)

                # 5. ⏱️ 1초 주기를 맞추기 위한 정밀 딜레이 계산
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, 1.0 - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n🛑 사용자 강제 종료! 실시간 추론을 안전하게 정지합니다.")
            print("🎉 수고하셨습니다 캡틴! 얼른 Jongbal 하세요!!!")



def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--vehicle-name", type=str, default="EgoCar")
    parser.add_argument("--lidar-name", type=str, default="Lidar1")
    parser.add_argument("--settings-path", type=str, default="~/Documents/AirSim/settings.json")
    parser.add_argument(
        "--cam-names",
        nargs="+",
        type=str,
        default=[
            "CAM_FRONT", 
            "CAM_FRONT_LEFT", 
            "CAM_FRONT_RIGHT", 
            "CAM_BACK", 
            "CAM_BACK_LEFT", 
            "CAM_BACK_RIGHT"
        ],
        help="사용할 카메라 이름 리스트 (띄어쓰기로 구분)"
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, opts = parse_args()
    evaluator = BEVFusionEvaluator(args, opts)
    evaluator.run()