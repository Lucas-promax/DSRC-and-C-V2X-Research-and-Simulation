import sys
import os

# --- Fix: force UTF-8 output to avoid garbled console text (esp. on Windows) ---
try:
    sys.stdout.reconfigure(encoding="utf-8") # type: ignore
    sys.stderr.reconfigure(encoding="utf-8") # type: ignore
except Exception:
    # Python < 3.7 or some environments
    os.environ["PYTHONIOENCODING"] = "utf-8"

import cv2
import socket
import json
import time
import threading
import numpy as np
from ultralytics import YOLO  # type: ignore


# ===================== 全局配置 =====================
# 感知模块配置
MODEL_PATH = "yolov5s.pt"
VIDEO_PATH = "traffic_night.mp4"  # 替换为你的视频路径
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 6, 7]
CLASS_NAMES = {}

# C-V2X通信配置
RSU_HOST = "127.0.0.1"
RSU_PORT = 8080
DEVICE_ID = "vehicle_001"
SEND_INTERVAL = 0.5

# 可视化配置
WINDOW_TITLE = "C-V2X车路协同感知信息可视化仿真系统"
LOG_MAX_LINES = 10  # 日志最大显示行数
traffic_logs = []  # 通信日志列表


# ===================== 感知模块 =====================
def init_perception():
    global CLASS_NAMES
    model = YOLO(MODEL_PATH)
    CLASS_NAMES = model.names
    return model


def detect_and_pack_data(model, frame):
    results = model(frame, conf=0.5)
    detect_info = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in TRAFFIC_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_name = CLASS_NAMES[cls_id]

                    obj_info = {
                        "class_name": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "device_id": DEVICE_ID,
                    }
                    detect_info.append(obj_info)

    cv2x_data = {
        "timestamp": str(cv2.getTickCount()),
        "device_id": DEVICE_ID,
        "detect_objects": detect_info,
        "message_type": "perception_data",
    }
    cv2x_json = json.dumps(cv2x_data, ensure_ascii=False)
    vis_frame = results[0].plot()

    return vis_frame, cv2x_json, detect_info


# ===================== C-V2X通信模块 =====================
def send_data_to_rsu(cv2x_json: str):
    """
    Scheme A fix:
    1) console output forced to UTF-8 (top of file)
    2) log text in window uses cv2.putText -> avoid Chinese here; use ASCII/English
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RSU_HOST, RSU_PORT))
            s.sendall(cv2x_json.encode("utf-8"))

        obj_cnt = len(json.loads(cv2x_json).get("detect_objects", []))
        log = f"[{time.strftime('%H:%M:%S')}] {DEVICE_ID} -> RSU: sent OK (objects: {obj_cnt})"
    except Exception as e:
        log = f"[{time.strftime('%H:%M:%S')}] {DEVICE_ID} -> RSU: sent FAILED ({type(e).__name__}: {e})"

    traffic_logs.append(log)
    if len(traffic_logs) > LOG_MAX_LINES:
        traffic_logs.pop(0)

    print(log)


# ===================== 交通场景可视化 =====================
def draw_traffic_scene(detect_info):
    # 创建空白画布（道路场景）
    scene_width, scene_height = 640, 480
    scene = np.ones((scene_height, scene_width, 3), dtype=np.uint8) * 20  # 深灰色背景

    # 绘制道路
    cv2.rectangle(scene, (100, 50), (540, 430), (100, 100, 100), -1)  # 灰色路面
    cv2.line(scene, (320, 50), (320, 430), (255, 255, 0), 2)  # 黄色分隔线
    cv2.putText(
        scene,
        "Road Scene (C-V2X Cooperative Perception)",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # 绘制车载设备（自身）
    car_x, car_y = 320, 350
    cv2.rectangle(scene, (car_x - 30, car_y - 20), (car_x + 30, car_y + 20), (0, 255, 0), -1)
    cv2.putText(
        scene,
        DEVICE_ID,
        (car_x - 40, car_y - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    # 绘制检测到的目标
    for obj in detect_info:
        # 将视频帧坐标映射到场景画布（简化映射）
        obj_cls = obj["class_name"]
        bbox = obj["bbox"]

        # 简化：用目标框的中心点映射到场景
        obj_center_x = int((bbox[0] + bbox[2]) / 2)
        obj_center_y = int((bbox[1] + bbox[3]) / 2)

        # 场景中的目标位置（缩放映射）
        scene_x = int(100 + (obj_center_x / frame_width) * 440)
        scene_y = int(50 + (obj_center_y / frame_height) * 380)

        # 根据类别绘制不同颜色和形状
        if obj_cls == "person":
            cv2.circle(scene, (scene_x, scene_y), 10, (0, 0, 255), -1)
            cv2.putText(
                scene,
                "Person",
                (scene_x - 20, scene_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
        elif obj_cls in ["car", "bus", "truck"]:
            cv2.rectangle(scene, (scene_x - 15, scene_y - 10), (scene_x + 15, scene_y + 10), (255, 0, 0), -1)
            cv2.putText(
                scene,
                "Car",
                (scene_x - 15, scene_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        elif obj_cls in ["bicycle", "motorcycle"]:
            cv2.polylines(
                scene,
                [np.array([[scene_x, scene_y - 10], [scene_x - 10, scene_y + 5], [scene_x + 10, scene_y + 5]])],
                True,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                scene,
                "Bike",
                (scene_x - 15, scene_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

    return scene


# ===================== 日志可视化 =====================
def draw_log_window():
    log_width, log_height = 640, 200
    log_window = np.ones((log_height, log_width, 3), dtype=np.uint8) * 40
    cv2.putText(
        log_window,
        "C-V2X Data Transmission Log",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # 绘制日志（注意：这里用 cv2.putText，不要放中文，已在 send_data_to_rsu 改成英文）
    for i, log in enumerate(traffic_logs):
        y = 60 + i * 15
        cv2.putText(log_window, log, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return log_window


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 初始化感知模型
    model = init_perception()

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    global frame_width, frame_height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化发送时间
    last_send_time = time.time()

    # 创建窗口
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 感知模块
        vis_frame, cv2x_json, detect_info = detect_and_pack_data(model, frame)

        # 2. 定时发送C-V2X数据
        current_time = time.time()
        if current_time - last_send_time >= SEND_INTERVAL:
            # 开启线程发送数据，不阻塞可视化
            send_thread = threading.Thread(target=send_data_to_rsu, args=(cv2x_json,))
            send_thread.daemon = True
            send_thread.start()
            last_send_time = current_time

        # 3. 绘制各窗口
        traffic_scene = draw_traffic_scene(detect_info)  # 交通场景
        log_window = draw_log_window()  # 通信日志

        # 拼接所有窗口（2行2列）
        row1 = np.hstack((cv2.resize(vis_frame, (640, 480)), traffic_scene))
        row2 = np.hstack((log_window, np.ones((200, 640, 3), dtype=np.uint8) * 40))  # 空白窗口（可扩展）
        combined_window = np.vstack((row1, row2))

        # 显示综合窗口
        cv2.imshow(WINDOW_TITLE, combined_window)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("Simulation closed.")
