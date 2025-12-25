#!/usr/bin/env bash
set -e

echo "=== DSRC vs C-V2X Simulation: 3 Representative Scenarios ==="
echo

# =========================
# Case 1: 低密度 / 轻负载
# =========================
# 场景说明：
# - 模拟车辆稀疏的高速路或郊区道路
# - 信道竞争较少，DSRC 和 C-V2X 性能应接近
# - 用作“基线对照实验”
echo "[Case 1] Low density / light traffic"
python ./simulation/sim_dsrc_cv2x.py \
  --vehicles 80 \
  --seconds 20 \
  --msg_rate 10 \
  --seed 1

echo
sleep 1

# =========================
# Case 2: 中高密度 / 常见城市道路
# =========================
# 场景说明：
# - 模拟城市主干道或高峰期车流
# - 信道开始出现明显竞争
# - 预期：C-V2X 的 PDR 和延迟开始优于 DSRC
echo "[Case 2] Medium–high density / typical urban traffic"
python ./simulation/sim_dsrc_cv2x.py \
  --vehicles 180 \
  --seconds 25 \
  --msg_rate 10 \
  --seed 7

echo
sleep 1
