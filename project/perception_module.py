import cv2
import json
from ultralytics import YOLO #type: ignore
import numpy as np

# 1. 初始化YOLOv5模型（轻量化yolov5s，适合实时检测）
model = YOLO('yolov5s.pt')

# 定义交通相关目标类别（YOLOv5的coco数据集类别）
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 6, 7]  # 对应：人、自行车、汽车、摩托车、公交车、卡车、自行车
CLASS_NAMES = model.names  # 获取类别名称映射

def detect_and_pack_data(frame):
    """
    对单帧图像进行检测，并封装为C-V2X传输格式
    :param frame: 输入视频帧
    :return: 检测结果可视化帧、封装后的JSON数据
    """
    # 2. 执行目标检测
    results = model(frame, conf=0.5)  # conf=0.5：置信度阈值，过滤低置信度目标
    
    # 3. 提取检测结果
    detect_info = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                # 只保留交通相关目标
                if cls_id in TRAFFIC_CLASSES:
                    # 提取目标信息
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 目标框坐标
                    conf = float(box.conf[0])  # 置信度
                    cls_name = CLASS_NAMES[cls_id]  # 类别名称
                    
                    # 存储单目标信息
                    obj_info = {
                        "class_name": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "device_id": "vehicle_001"  # 模拟车载设备ID
                    }
                    detect_info.append(obj_info)
    
    # 4. 封装为C-V2X传输的JSON数据
    cv2x_data = {
        "timestamp": str(cv2.getTickCount()),  # 时间戳（模拟）
        "device_id": "vehicle_001",
        "detect_objects": detect_info,
        "message_type": "perception_data"  # 消息类型：感知数据
    }
    cv2x_json = json.dumps(cv2x_data, ensure_ascii=False)
    
    # 5. 绘制检测框，生成可视化帧
    vis_frame = results[0].plot()  # YOLOv5自带绘制功能
    
    return vis_frame, cv2x_json

# 测试感知模块
if __name__ == "__main__":
    # 读取视频（替换为你的视频路径，0为电脑摄像头）
    cap = cv2.VideoCapture("traffic_night.mp4")  # 替换为你的视频路径
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 执行检测和数据封装
        vis_frame, cv2x_json = detect_and_pack_data(frame)
        
        # 显示检测结果
        cv2.imshow("YOLOv5 Traffic Detection (Dark Environment)", vis_frame)
        # 打印封装后的C-V2X数据（模拟传输数据）
        print("C-V2X传输数据：", cv2x_json[:200], "...")
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()