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
        """Initialization and configuration"""
        dist.init()
        self.args = args
        
        # Load and merge configurations
        configs.load(args.config, recursive=True)
        configs.update(opts)
        self.cfg = Config(self._recursive_eval(configs), filename=args.config)

        # GPU configuration
        torch.backends.cudnn.benchmark = self.cfg.cudnn_benchmark
        torch.cuda.set_device(dist.local_rank())

        # Initialize dataloader and model
        self.model = self._build_and_load_model()
        print(self.model)
        
        # AirSimulator configuration
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.armDisarm(True, vehicle_name=args.vehicle_name)
        self.sensors_config = self._load_sensor_configs(args.settings_path, args.vehicle_name, args.lidar_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # -------------- SIMULATOR RELATED -------------------
    
    def _load_sensor_configs(self, settings_path, vehicle_name, lidar_name):
        settings_path = os.path.expanduser(settings_path)
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Error: {settings_path} not found.")

        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            
        drone_settings = settings.get("Vehicles", {}).get(vehicle_name, {})
        cameras = drone_settings.get("Cameras", {})
        
        configs = {}
        
        # 1. Extract camera settings (Invert Y, Z coordinates and Yaw, Roll signs)
        for cam_name, cam_info in cameras.items():
            configs[cam_name] = {
                "t": [cam_info.get("X", 0.0), -cam_info.get("Y", 0.0), -cam_info.get("Z", 0.0)],
                "r": [cam_info.get("Pitch", 0.0), -cam_info.get("Roll", 0.0), -cam_info.get("Yaw", 0.0)]
            }
            
        # 2. Extract LiDAR settings (Invert signs here as well)
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
            # 1. Convert the returned 1D byte data to an 8-bit integer numpy array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            
            # 2. Reshape to match the original image resolution (height, width) and 3 channels (RGB)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            
            # 3. Append to the list
            images.append(img_rgb)
            
        return images
    
    @property
    def lidar(self):
        accumulated_points = []
        last_timestamp = 0  # Variable to store the timestamp of the last captured frame
        frames_collected = 0
        
        # Maximum retry limit to prevent infinite loops
        max_retries = 50 
        retries = 0
        
        # Repeat until exactly 10 frames with different timestamps are collected
        while frames_collected < 10 and retries < max_retries:
            lidar_data = self.client.getLidarData(self.args.lidar_name, vehicle_name=self.args.vehicle_name)
            
            # Collect only if data exists and the timestamp differs from the last captured frame
            if len(lidar_data.point_cloud) >= 3 and lidar_data.time_stamp != last_timestamp:
                points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
                
                # Coordinate axis inversion
                points[:, 1] = -points[:, 1]
                points[:, 2] = -points[:, 2]
                
                accumulated_points.append(points)
                
                # Update the timestamp
                last_timestamp = lidar_data.time_stamp
                frames_collected += 1
            
            # Wait briefly (0.005s) to allow the engine to process the next tick
            time.sleep(0.005)
            retries += 1
            
        if not accumulated_points:
            return None
            
        # Merge the collected frames
        merged_points = np.vstack(accumulated_points)
        
        # Add padding to match the BEVFusion model format
        padding = np.zeros((merged_points.shape[0], 2), dtype=np.float32)
        return np.hstack([merged_points, padding])
    
    
    def _get_transformation_matrix(self, t, r):
        """
        Takes [X, Y, Z] and [Pitch, Roll, Yaw] (Degrees) and returns a 4x4 Transformation Matrix (Tensor).
        """
        # 1. Convert Degrees to Radians
        pitch, roll, yaw = map(math.radians, r)
        
        # 2. Calculate Rotation Matrix (Based on standard Z-Y-X Euler angles)
        # Rotate along Roll(X), Pitch(Y), Yaw(Z) axes according to the AirSim environment
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
        
        # 3. Assemble 4x4 Transformation Matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R        # Top-left 3x3 is rotation
        T[:3, 3] = t         # Top-right 3x1 is translation
        
        return torch.tensor(T, dtype=torch.float32, device=self.device)
    
    
    def _get_camera_intrinsics(self, cam_name, img_width=704, img_height=256, orig_width=800, orig_height=450):
        cam_info = self.client.simGetCameraInfo(cam_name, vehicle_name=self.args.vehicle_name)
        fov_rad = math.radians(cam_info.fov)
        
        # 1. Focal length based on original resolution (800x450)
        fx_orig = orig_width / (2.0 * math.tan(fov_rad / 2.0))
        fy_orig = fx_orig 
        
        # 2. Scale according to the resize ratio
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
        N_cams = len(self.args.cam_names) # 6 cameras
        
        # Get transformed images
        transform = transforms.Compose([
            transforms.ToTensor(), # HWC -> CHW conversion and 0~1 scaling
            transforms.Resize((256, 704)), # Target resolution for BEVFusion
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensors = [transform(img) for img in current_images]
        batch_gpu['img'] = torch.stack(img_tensors, dim=0).unsqueeze(0).to(self.device)
        
        # Get transformed points
        batch_gpu['points'] = [torch.tensor(current_lidar, dtype=torch.float32).to(self.device)]
        
        
        # Get lidar to ego transformation
        lidar_cfg = self.sensors_config["LIDAR_TOP"]
        lidar2ego = self._get_transformation_matrix(lidar_cfg["t"], lidar_cfg["r"])
        batch_gpu['lidar2ego'] = lidar2ego.unsqueeze(0) # Store in [1, 4, 4] format
        
        
        # Get camera to ego, ego to camera, and lidar to camera transformations
        camera2ego_list = []
        lidar2camera_list = []
        camera2lidar_list = []
        
        body2optical = torch.tensor([
            [0, -1,  0,  0], # Optical X (Right) = -Body Y (Left)
            [0,  0, -1,  0], # Optical Y (Down) = -Body Z (Up)
            [1,  0,  0,  0], # Optical Z (Front) = Body X (Front)
            [0,  0,  0,  1]
        ], dtype=torch.float32, device=self.device)

        for cam_name in self.args.cam_names:
            cam_cfg = self.sensors_config[cam_name]
            cam2ego = self._get_transformation_matrix(cam_cfg["t"], cam_cfg["r"])
            camera2ego_list.append(cam2ego)
            
            ego2cam = torch.linalg.inv(cam2ego) 
            l2c_body = torch.matmul(ego2cam, lidar2ego) # Body-centric frame up to this point
            
            # Convert Body frame to Optical frame
            l2c_optical = torch.matmul(body2optical, l2c_body)
            lidar2camera_list.append(l2c_optical)
            
            c2l = torch.linalg.inv(l2c_optical)
            camera2lidar_list.append(c2l)

        # Stack into GPU tensor format [1, 6, 4, 4] 
        batch_gpu['camera2ego'] = torch.stack(camera2ego_list, dim=0).unsqueeze(0)
        batch_gpu['lidar2camera'] = torch.stack(lidar2camera_list, dim=0).unsqueeze(0)
        batch_gpu['camera2lidar'] = torch.stack(camera2lidar_list, dim=0).unsqueeze(0)
        
        # Camera Intrinsics 
        intrinsics_list = []
        for cam_name in self.args.cam_names:
            # Note: Calculate based on BEVFusion resize resolution (704x256)
            K = self._get_camera_intrinsics(cam_name, img_width=704, img_height=256)
            intrinsics_list.append(K)
            
        # Stack in [6, 4, 4] format, expand batch dimension to [1, 6, 4, 4]
        batch_gpu['camera_intrinsics'] = torch.stack(intrinsics_list, dim=0).unsqueeze(0)
        
        # (Intrinsics @ Lidar2Camera)
        batch_gpu['lidar2image'] = torch.matmul(batch_gpu['camera_intrinsics'], batch_gpu['lidar2camera'])
        
        
        # 4. Augmentation matrix (Identity matrix since there is no image/LiDAR rotation during inference)
        eye_4x4 = torch.eye(4, dtype=torch.float32, device=self.device)
        batch_gpu['img_aug_matrix'] = eye_4x4.view(1, 1, 4, 4).repeat(1, N_cams, 1, 1)
        batch_gpu['lidar_aug_matrix'] = eye_4x4.view(1, 4, 4)
        
        
        # 5. Metadata (Prevents errors when the model references timestamp or token internally)
        batch_gpu['metas'] = [{
            'timestamp': datetime.now().timestamp(),
            'token': 'airsim_token',
            'box_mode_3d': 0, # LiDAR box mode for MMDetection3D
            'box_type_3d': LiDARInstance3DBoxes
        }]
        
        batch_gpu['depths'] = torch.zeros(1, device='cuda')
        
        return batch_gpu
    
    
    # -------------------------- MODEL RELATED ---------------------------
    
    def _recursive_eval(self, obj, globals_dict=None):
        """Recursively evaluate and parse configurations"""
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
        """Build model and load weights"""
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
        """Extract prediction data"""
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
        """Direct memory visualization function for real-time AirSim data"""
        
        # 1. Camera visualization (Includes automatic save path generation)
        for k, image in enumerate(raw_images):
            out_path = os.path.join(self.args.out_dir, f"camera-{k}", f"{name}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # Match the canvas to the model projection matrix criteria (704x256)
            image_resized = cv2.resize(image, (704, 256))
            
            visualize_camera(
                out_path,
                image_resized, # Use image_resized instead of the original image
                bboxes=bboxes,
                labels=labels,
                transform=metas["lidar2image"][k],
                classes=self.cfg.object_classes,
            )

        # 2. LiDAR visualization
        # Extract directly from data["points"][0] as there is no DataContainer wrapper
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

        # 3. BEV Map visualization
        if masks is not None:
            out_path = os.path.join(self.args.out_dir, "map", f"{name}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            visualize_map(
                out_path,
                masks,
                classes=self.cfg.map_classes,
            )

    def run(self):
        """Pipeline that fetches real-time AirSim data every second for inference and visualization"""
        print("Starting real-time AirSim BEVFusion inference. (Press Ctrl+C to exit)")

        try:
            while True:
                loop_start = time.time()

                # 1. Capture current frame (Caching to prevent duplicate API calls)
                current_images = self.images
                current_lidar = self.lidar

                if current_lidar is None: # Skip if LiDAR is not fully loaded
                    print("Waiting for LiDAR response...")
                    time.sleep(0.1)
                    continue

                # 2. Assemble tensor batch for the deep learning model
                data = self.get_batch_gpu(current_images, current_lidar)
                
                # Extract metadata and copy projection matrix (lidar2image) for visualization
                metas = data["metas"][0]
                metas["lidar2image"] = data["lidar2image"][0].cpu().numpy()
                name = f"{metas['timestamp']}-{metas['token']}"
                
                print(f"[AirSim] Processing frame: {name}")
                
                if 'points' in data and len(data['points']) > 0:
                    print("LiDAR Shape:", data['points'][0].shape)
                    if data['points'][0].shape[0] > 0:
                        # Check the range by verifying the min and max values of X, Y, Z coordinates
                        print("X min/max:", data['points'][0][:, 0].min().item(), "/", data['points'][0][:, 0].max().item())
                        print("Y min/max:", data['points'][0][:, 1].min().item(), "/", data['points'][0][:, 1].max().item())
                        print("Z min/max:", data['points'][0][:, 2].min().item(), "/", data['points'][0][:, 2].max().item())
                    else:
                        print("Point cloud count is 0.")

                # 3. BEVFusion inference (Always pred mode since there is no GT in real-time)
                with torch.inference_mode():
                    outputs = self.model(**data)
                bboxes, labels, masks = self._extract_predictions(outputs)
                
                if bboxes is not None and len(bboxes) > 0:
                    print(f"\n[Frame: {name}] Detected objects count: {len(bboxes)}")
                    
                    # Convert the tensor inside LiDARInstance3DBoxes to a numpy array
                    # Shape: [N, 9] -> (x, y, z, x_size, y_size, z_size, yaw, vx, vy)
                    box_data = bboxes.tensor.cpu().numpy()
                    
                    for i in range(len(bboxes)):
                        # Position (Center X, Y, Z)
                        x, y, z = box_data[i][0:3]
                        # Size (Length, Width, Height)
                        l, w, h = box_data[i][3:6]
                        # Rotation angle (Radians)
                        yaw = box_data[i][6]
                        
                        # Class name matching (e.g., Car, Pedestrian, etc.)
                        class_name = self.cfg.object_classes[labels[i]] if labels is not None else "Unknown"
                        
                        # Print nicely formatted output
                        print(f"  [{i+1}] {class_name:^10} | "
                              f"Position(x,y,z): ({x:>6.2f}, {y:>6.2f}, {z:>6.2f}) | "
                              f"Size(l,w,h): ({l:>5.2f}, {w:>5.2f}, {h:>5.2f}) | "
                              f"Angle: {yaw:>5.2f} rad")
                else:
                    print(f"\n[Frame: {name}] No objects detected.")

                # 4. Visualization and image saving
                self._visualize_realtime(data, metas, name, bboxes, labels, masks, current_images)

                # 5. Precise delay calculation to match the 1-second cycle
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, 1.0 - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nUser termination detected. Stopping real-time inference safely.")
            print("Execution finished.")



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
        help="List of camera names to use (separated by space)"
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, opts = parse_args()
    evaluator = BEVFusionEvaluator(args, opts)
    evaluator.run()