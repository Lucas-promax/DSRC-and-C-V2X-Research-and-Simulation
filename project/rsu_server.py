import socket
import sys
import io

# 尝试将标准输出设置为 UTF-8，避免 Windows 终端显示中文为问号
try:
    sys.stdout.reconfigure(encoding='utf-8') #type: ignore
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = '127.0.0.1'
PORT = 8080

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)
    print(f'RSU server listening on {HOST}:{PORT}')
    try:
        while True:
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                data = b''
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                def try_decode(b: bytes):
                    # 尝试多种常见编码，优先 utf-8，再尝试系统默认 cp936（GBK）等
                    enc_candidates = ['utf-8', 'cp936', 'gbk', 'cp1252', 'latin1']
                    for enc in enc_candidates:
                        try:
                            s = b.decode(enc)
                        except Exception:
                            continue
                        # 如果解码后仍然包含替换字符，尝试下一种编码
                        if '\ufffd' in s:
                            continue
                        return s, enc
                    # 最后使用 replace 以避免抛异常
                    return b.decode('utf-8', errors='replace'), 'utf-8(replace)'

                decoded, used_enc = try_decode(data)
                # 同时打印原始 bytes 的 repr 和部分 hex，便于排查编码或传输问题
                hex_preview = data[:64].hex()
                print(f"Connected by {addr} | decoded with: {used_enc}", flush=True)
                print('Received (decoded):', decoded, flush=True)
                print('Received (repr bytes):', repr(data), flush=True)
                print('Received (hex preview):', hex_preview, flush=True)
    except KeyboardInterrupt:
        print('RSU server stopped')