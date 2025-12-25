import socket
import json
import threading

# 路侧端配置
HOST = "127.0.0.1"  # 本地回环地址（模拟RSU部署地址）
PORT = 8080  # 监听端口
BUFFER_SIZE = 1024 * 4  # 数据缓冲区大小

# 存储所有车载设备的感知数据
vehicle_data_list = []

def handle_vehicle_connection(client_socket, client_addr):
    """
    处理单个车载设备的连接（TCP）
    """
    print(f"[RSU] 车载设备 {client_addr} 已连接")
    
    while True:
        try:
            # 接收车载端数据
            data = client_socket.recv(BUFFER_SIZE)
            if not data:
                break
            
            # 解析C-V2X JSON数据
            cv2x_json = data.decode("utf-8")
            cv2x_data = json.loads(cv2x_json)
            
            # 存储车载设备数据
            vehicle_data_list.append(cv2x_data)
            print(f"[RSU] 接收来自 {cv2x_data['device_id']} 的感知数据：")
            print(f"  - 检测到 {len(cv2x_data['detect_objects'])} 个交通目标")
            print(f"  - 时间戳：{cv2x_data['timestamp']}")
            print(f"  - 已汇总 {len(vehicle_data_list)} 个车载设备数据\n")
        
        except Exception as e:
            print(f"[RSU] 与 {client_addr} 连接异常：{e}")
            break
    
    client_socket.close()
    print(f"[RSU] 车载设备 {client_addr} 断开连接")

def start_rsu():
    """
    启动路侧端（RSU）
    """
    # 创建TCP socket（可靠传输，适合感知数据传输）
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)  # 最大监听5个车载设备
        print(f"[RSU] 路侧端已启动，监听 {HOST}:{PORT}...")
        
        while True:
            # 等待车载设备连接
            client_socket, client_addr = server_socket.accept()
            # 开启线程处理每个车载设备（支持多设备同时连接）
            thread = threading.Thread(target=handle_vehicle_connection, args=(client_socket, client_addr))
            thread.daemon = True
            thread.start()

if __name__ == "__main__":
    try:
        start_rsu()
    except KeyboardInterrupt:
        print("\n[RSU] 路侧端已关闭")