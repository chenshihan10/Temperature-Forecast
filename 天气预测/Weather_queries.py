import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import threading

# — 配置中文字体，确保中文正常显示 —
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用于正常显示负号

# — 接口配置 —
API_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
API_KEY = "e8728b164f0db0882e80336ccf165182"    # 高德API Key
CITY_ADCODE = "110000"
WINDOW = 12   # 窗口长度（12 个点 = 1 小时）
INTERVAL_MS = 5 * 60 * 1000  # 更新间隔：5 分钟（毫秒）

# — 全局状态 —
data_buf = []
scaler = MinMaxScaler()
# LSTM 模型结构，增加了 Dropout 层防止过拟合
model = Sequential([
    LSTM(64, input_shape=(WINDOW-1, 3), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(2)
])
model.compile("adam", "mse")

# — 获取实时天气函数 —
def fetch():
    r = requests.get(API_URL, params={"key": API_KEY, "city": CITY_ADCODE, "extensions": "base"}, timeout=5)
    r.raise_for_status()
    live = r.json()['lives'][0]
    return float(live['temperature']), float(live['humidity']), float(live['windspeed'])

# — Matplotlib 可视化设置 —
fig, ax = plt.subplots()
line_obs, = ax.plot([], [], label="观测温度 (℃)")
line_pred, = ax.plot([], [], label="预测温度 (℃)")
ax.set_xlabel("时间步（每步5分钟）")
ax.set_ylabel("温度 (℃)")
ax.set_title("北京市未来1小时温度预测")
ax.legend()

# — 停止控制 —
stop_event = threading.Event()

def on_key(event):
    if event.key in ('enter', 'q', 'escape'):
        stop_event.set()

fig.canvas.mpl_connect('key_press_event', on_key)

# — 初始化缓冲区：首次拉取并填充基准数据 —
def init_data():
    try:
        t0, h0, ws0 = fetch()
        print(f"初始基准：温度={t0}℃, 湿度={h0}%, 风速={ws0}m/s")
        return [[t0, h0, ws0] for _ in range(WINDOW)]
    except Exception as e:
        print(f"无法获取初始天气：{e}")
        return [[0, 0, 0] for _ in range(WINDOW)]

data_buf = init_data()

# — 动画更新函数 —
def update(frame):
    if stop_event.is_set():
        ani.event_source.stop()
        plt.close(fig)
        return

    # 采集新观测并更新缓冲
    t, h, ws = fetch()
    data_buf.append([t, h, ws])
    if len(data_buf) > WINDOW:
        data_buf.pop(0)

    # 清除旧图，重新绘制
    ax.clear()
    ax.set_xlabel("时间步（每步5分钟）")
    ax.set_ylabel("温度 (℃)")
    ax.set_title("北京市未来1小时温度预测")

    # 绘制观测温度
    obs = np.array(data_buf)[:, 0]
    ax.plot(obs, label="观测温度 (℃)")

    # 当缓冲满时训练并预测
    if len(data_buf) == WINDOW:
        arr = np.array(data_buf)
        scaled = scaler.fit_transform(arr)
        X = scaled[:-1].reshape(1, WINDOW-1, 3)  # 使用温度、湿度、风速作为特征
        y = scaled[1:].reshape(1, WINDOW-1, 3)[:, -1]
        model.fit(X, y, epochs=5, verbose=0)

        # 滚动预测
        seq = X.copy()
        preds = []
        for _ in range(WINDOW):
            p = model.predict(seq, verbose=0)
            preds.append(p[0, 0])
            seq = np.concatenate([seq[:, 1:, :], p.reshape(1, 1, 2)], axis=1)

        # 反归一化处理
        preds = scaler.inverse_transform(np.column_stack((preds, np.zeros(WINDOW), np.zeros(WINDOW))))[:, 0]
        ax.plot(preds, '--', label="预测温度 (℃)")

    ax.legend()

# — 启动动画 —
ani = FuncAnimation(fig, update, interval=INTERVAL_MS)
plt.show()
