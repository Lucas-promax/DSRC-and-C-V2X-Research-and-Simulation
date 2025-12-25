import os
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson

# ====== 你要求的根目录 ======
base_dir = "./visiual"
os.makedirs(base_dir, exist_ok=True)

CSV_PATH = "v2x.csv"
CSV_PATH = os.path.join(base_dir, CSV_PATH)
OUT_HTML = os.path.join(base_dir, "v2x_vehicle_time_map.html")

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["vehicle_id"] = df["vehicle_id"].astype(str)
df = df.sort_values(["timestamp", "vehicle_id"])

vehicle_ids = sorted(df["vehicle_id"].unique().tolist())

# 地图中心
center_lat = float(df["gps_latitude"].mean())
center_lon = float(df["gps_longitude"].mean())

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles="OpenStreetMap",
    control_scale=True
)

# ====== 构造单个 GeoJSON（关键：避免多个时间轴冲突）======
features = []
for _, r in df.iterrows():
    vid = r["vehicle_id"]
    t = r["timestamp"].isoformat()

    # 点击显示的完整信息（你 CSV 的字段都放进来）
    popup_html = f"""
    <div style="font-family:system-ui;font-size:12px;max-width:280px;">
      <div style="font-size:13px;font-weight:800;">Vehicle {vid}</div>
      <div style="opacity:.8;">{r["timestamp"]}</div>
      <hr style="margin:6px 0;border:0;border-top:1px solid #e5e7eb;">
      <b>GPS</b>: ({float(r["gps_latitude"]):.6f}, {float(r["gps_longitude"]):.6f})<br>
      <b>Latency</b>: {int(r["latency_ms"])} ms<br>
      <b>PDR</b>: {float(r["packet_delivery_ratio"]):.3f}<br>
      <b>Drop Rate</b>: {float(r["packet_drop_rate"]):.3f}<br>
      <b>Throughput</b>: {int(r["throughput_kbps"])} kbps<br>
      <hr style="margin:6px 0;border:0;border-top:1px solid #e5e7eb;">
      <b>Lidar</b>: {int(r["lidar_points"])}<br>
      <b>Radar</b>: {int(r["radar_objects"])}<br>
      <b>Camera</b>: {int(r["camera_objects"])}<br>
      <b>Obstacle Acc</b>: {float(r["obstacle_detection_accuracy"]):.2f}<br>
      <b>Decision Acc</b>: {float(r["decision_accuracy"]):.2f}<br>
      <b>Collision</b>: {int(r["collision_detected"])}
    </div>
    """

    # 颜色：碰撞红色，否则青色
    is_col = int(r["collision_detected"]) == 1
    color = "#ff4d4f" if is_col else "#22d3ee"

    # ✅ 给每个点一个 CSS class：vid-xxxx，用于“下拉筛选隐藏其它车”
    class_name = f"vid-{vid}"

    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(r["gps_longitude"]), float(r["gps_latitude"])]},
        "properties": {
            "time": t,
            "popup": popup_html,
            "icon": "circle",
            "iconstyle": {
                "fillColor": color,
                "fillOpacity": 0.85,
                "stroke": True,
                "color": "#0b1220",
                "weight": 1,
                "radius": 6,
                "className": class_name,   # ⭐ 关键
            },
        },
    })

geojson = {"type": "FeatureCollection", "features": features}

# ====== 时间轴（只保留当前 1 秒窗口，避免满屏点）======
TimestampedGeoJson(
    geojson,
    period="PT1S",
    duration="PT1S",          # ⭐ 只显示当前窗口
    transition_time=200,
    add_last_point=False,
    auto_play=False,
    loop=False,
    time_slider_drag_update=True,
    date_options="YYYY-MM-DD HH:mm:ss",
).add_to(m)

# ====== 注入：vehicle_id 下拉筛选控件（纯前端，真实可用）======
options_html = "\n".join([f'<option value="{vid}">{vid}</option>' for vid in vehicle_ids])

inject = f"""
<style>
/* 一个“看起来像产品”的控件 */
.v2x-filter {{
  position: fixed;
  top: 14px;
  right: 14px;
  z-index: 9999;
  background: rgba(15, 23, 42, 0.92);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 10px 12px;
  color: #e5e7eb;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}}
.v2x-filter label {{
  font-size: 12px;
  opacity: 0.85;
  display: block;
  margin-bottom: 6px;
}}
.v2x-filter select {{
  width: 220px;
  background: rgba(255,255,255,0.06);
  color: #e5e7eb;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 10px;
  padding: 8px 10px;
  outline: none;
}}
.v2x-filter .hint {{
  margin-top: 6px;
  font-size: 11px;
  opacity: 0.7;
}}
</style>

<div class="v2x-filter">
  <label>Filter vehicle_id</label>
  <select id="vidSelect">
    <option value="ALL">ALL</option>
    {options_html}
  </select>
  <div class="hint">拖动时间轴；点位可点击看详细信息</div>
</div>

<style id="vidFilterStyle"></style>

<script>
(function() {{
  function setFilter(vid) {{
    var styleTag = document.getElementById("vidFilterStyle");
    if (!styleTag) return;

    if (vid === "ALL") {{
      styleTag.innerHTML = "";
      return;
    }}

    // Leaflet circleMarker 渲染成 SVG path（className 会挂在 path 上）
    // 只显示当前选中 vehicle 的点，其它点隐藏
    styleTag.innerHTML = `
      path.leaflet-interactive:not(.vid-${{vid}}) {{ display: none !important; }}
    `;
  }}

  var sel = document.getElementById("vidSelect");
  sel.addEventListener("change", function() {{
    setFilter(this.value);
  }});

  // 默认不过滤
  setFilter("ALL");
}})();
</script>
"""

m.get_root().html.add_child(folium.Element(inject)) # type: ignore

m.save(OUT_HTML)
print(f"✅ Saved: {OUT_HTML}")
