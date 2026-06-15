import airsim
import time
import sys
import tty
import termios
import select  
import datetime
import numpy as np

def get_key(timeout=0.1):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        r, w, e = select.select([sys.stdin], [], [], timeout)
        if r:
            ch = sys.stdin.read(1)
        else:
            ch = ''  
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

print("🔗 Connecting to AirSim...")
client = airsim.CarClient() 
client.confirmConnection()
client.enableApiControl(True)

car_controls = airsim.CarControls()

print("\n" + "="*45)
print(" 🏎️ SOTA Custom Car Controller V3 (w/ Position Tracker) 🏎️")
print("="*45)
print(" [W/S] Throttle/Reverse| [A/D] Steering Left/Right")
print(" [Space] Brake")
print(" -----------------------------------------")
print(" [P] Extract LiDAR Data & Save PCD")
print("="*45)
print(" [C] Exit Program\n")

# 🌟 [Core 1] Record initial time before loop starts
last_print_time = time.time()

while True:
    key = get_key(timeout=0.05).lower()

    if key == 'c':
        print("\n🛑 Car stopped! Exiting controller.")
        break
        
    car_controls.steering = 0.0
    car_controls.throttle = 0.0
    car_controls.brake = 0.0
    car_controls.is_manual_gear = False

    if key == 'w':
        car_controls.throttle = 1.0  
    elif key == 's':
        car_controls.throttle = -1.0 
        car_controls.is_manual_gear = True
        car_controls.manual_gear = -1
    elif key == 'a':
        car_controls.steering = -1.0 
        car_controls.throttle = 0.85  
    elif key == 'd':
        car_controls.steering = 1.0  
        car_controls.throttle = 0.85  
    elif key == ' ':
        car_controls.brake = 1.0     
    
    client.setCarControls(car_controls)

    # 🌟 [Core 2] Print position every 1 second without interrupting control loop!
    current_time = time.time()
    if current_time - last_print_time >= 1.0:
        state = client.getCarState()
        pos = state.kinematics_estimated.position
        # Adding \r at the front updates the numbers in place without spamming newlines in the terminal!
        sys.stdout.write(f"\r📍 [Current Position] X: {pos.x_val:8.2f} | Y: {pos.y_val:8.2f} | Z: {pos.z_val:8.2f}   ")
        sys.stdout.flush()
        last_print_time = current_time

    # --- LiDAR Data Saving Logic ---
    if key == 'p':
        print("\n\n📡 Saving LiDAR data as PCD format...")
        
        accumulated_points = []
        
        for _ in range(10):
            lidar_data = client.getLidarData(lidar_name="Lidar1", vehicle_name="EgoCar")
            if len(lidar_data.point_cloud) >= 3:
                points = np.array(lidar_data.point_cloud, dtype=np.float32)
                points = np.reshape(points, (int(points.shape[0] / 3), 3))
                accumulated_points.append(points)
            time.sleep(0.1) 
        
        if not accumulated_points:
            print("😱 No points found! Please try in an area with a floor/ground.")
        else:
            final_points = np.vstack(accumulated_points)
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"Lidar_{now}.pcd"
            
            with open(filename, 'w') as f:
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\n")
                f.write("FIELDS x y z\n")
                f.write("SIZE 4 4 4\n")
                f.write("TYPE F F F\n")
                f.write("COUNT 1 1 1\n")
                f.write(f"WIDTH {final_points.shape[0]}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {final_points.shape[0]}\n")
                f.write("DATA ascii\n")
                
                for i in range(final_points.shape[0]):
                    f.write(f"{final_points[i][0]} {final_points[i][1]} {final_points[i][2]}\n")
            
            print(f"✅ Save complete! Filename: {filename} (Points: {final_points.shape[0]})\n")
            
        # Reset the timer to count 1 second again for the next output since pressing P caused a delay!
        last_print_time = time.time()

client.enableApiControl(False)