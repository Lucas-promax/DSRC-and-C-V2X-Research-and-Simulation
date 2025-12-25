import socket
import time
from perception_module import detect_and_pack_data  # 导入第一步的感知模块
import cv2
import argparse

# 命令行参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--ID",type=str,default="vehicle_001",help="车载设备ID")
args = parser.parse_args()


# 车载端配置
RSU_HOST = "127.0.0.1"  # 路侧端地址
RSU_PORT = 8080  # 路侧端端口
DEVICE_ID = args.ID  # 车载设备ID（可修改为vehicle_002、003模拟多设备）

def send_perception_data_to_rsu(frame):
    """
    获取感知数据并发送给路侧端
    """
    # 1. 获取YOLOv5检测并封装的数据
    vis_frame, cv2x_json = detect_and_pack_data(frame)
    
    # 2. 创建TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            # 连接路侧端
            client_socket.connect((RSU_HOST, RSU_PORT))
            # 发送数据
            client_socket.sendall(cv2x_json.encode("utf-8"))
            print(f"[{DEVICE_ID}] 感知数据已发送给RSU\n")
        except ConnectionRefusedError:
            print(f"[{DEVICE_ID}] 无法连接到RSU，请先启动路侧端")
    
    return vis_frame

if __name__ == "__main__":
    # 读取视频（替换为你的视频路径，0为摄像头）
    cap = cv2.VideoCapture("traffic_night.mp4")
    
    # 每隔0.5秒发送一次数据（模拟实时传输）
    send_interval = 0.5
    last_send_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 定时发送数据
        current_time = time.time()
        if current_time - last_send_time >= send_interval:
            vis_frame = send_perception_data_to_rsu(frame)
            last_send_time = current_time
        else:
            # 不发送数据时，也更新检测画面
            vis_frame, _ = detect_and_pack_data(frame)
        
        # 显示检测画面
        cv2.imshow(f"[{DEVICE_ID}] Traffic Detection", vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()