import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fs_analog = 1000
T = 1.0
t = np.arange(0, T, 1/fs_analog)
N = len(t)

f1_center = 50
f1_bandwidth = 20
A1 = 1.0

f2_center = 150
f2_bandwidth = 30

A2 = A1 * 10**(20/20)

print(f"有用信号参数：中心频率={f1_center}Hz，带宽={f1_bandwidth}Hz，幅度={A1}")
sa1 = A1 * np.sin(2 * np.pi * f1_center * t) * np.sin(np.pi * f1_bandwidth * t) / (np.pi * f1_bandwidth * t + 1e-10)

window = np.hanning(len(t))
sa1 = sa1 * window


print(f"干扰信号参数：中心频率={f2_center}Hz，带宽={f2_bandwidth}Hz，幅度={A2}")
sa2 = A2 * np.sin(2 * np.pi * f2_center * t) * np.sin(np.pi * f2_bandwidth * t) / (np.pi * f2_bandwidth * t + 1e-10)
sa2 = sa2 * window

xa = sa1 + sa2

freqs = fftfreq(N, 1/fs_analog)
Sa1_f = fft(sa1)
Sa2_f = fft(sa2)
Xa_f = fft(xa)

fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10))

axes1[0].plot(t, sa1, 'b', linewidth=1.5)
axes1[0].set_xlabel('时间 (s)')
axes1[0].set_ylabel('幅度')
axes1[0].set_title('有用信号 sa1(t) 时域波形')
axes1[0].grid(True, alpha=0.3)
axes1[0].set_xlim([0, 0.1])

axes1[1].plot(t, sa2, 'r', linewidth=1.5)
axes1[1].set_xlabel('时间 (s)')
axes1[1].set_ylabel('幅度')
axes1[1].set_title('干扰信号 sa2(t) 时域波形')
axes1[1].grid(True, alpha=0.3)
axes1[1].set_xlim([0, 0.1])

axes1[2].plot(t, xa, 'g', linewidth=1.5)
axes1[2].set_xlabel('时间 (s)')
axes1[2].set_ylabel('幅度')
axes1[2].set_title('合成信号 xa(t) = sa1(t) + sa2(t) 时域波形')
axes1[2].grid(True, alpha=0.3)
axes1[2].set_xlim([0, 0.1])

plt.tight_layout()

fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))

sa1_spectrum = np.abs(Sa1_f[:N//2])
freq_plot = freqs[:N//2]
axes2[0].plot(freq_plot, sa1_spectrum, 'b', linewidth=1.5)
axes2[0].set_xlabel('频率 (Hz)')
axes2[0].set_ylabel('幅度')
axes2[0].set_title('有用信号 sa1(t) 频谱')
axes2[0].grid(True, alpha=0.3)
axes2[0].set_xlim([0, 250])

sa2_spectrum = np.abs(Sa2_f[:N//2])
axes2[1].plot(freq_plot, sa2_spectrum, 'r', linewidth=1.5)
axes2[1].set_xlabel('频率 (Hz)')
axes2[1].set_ylabel('幅度')
axes2[1].set_title('干扰信号 sa2(t) 频谱')
axes2[1].grid(True, alpha=0.3)
axes2[1].set_xlim([0, 250])

xa_spectrum = np.abs(Xa_f[:N//2])
axes2[2].plot(freq_plot, xa_spectrum, 'g', linewidth=1.5)
axes2[2].set_xlabel('频率 (Hz)')
axes2[2].set_ylabel('幅度')
axes2[2].set_title('合成信号 xa(t) 频谱')
axes2[2].grid(True, alpha=0.3)
axes2[2].set_xlim([0, 250])

plt.tight_layout()
plt.show()


f_max = f2_center + f2_bandwidth/2
print(f"合成信号最高频率成分: {f_max:.2f} Hz")


fs = 400
print(f"选择采样频率: fs = {fs} Hz")


n = np.arange(0, T, 1/fs)
N_d = len(n)

s1 = A1 * np.sin(2 * np.pi * f1_center * n) * np.sin(np.pi * f1_bandwidth * n) / (np.pi * f1_bandwidth * n + 1e-10)
s1_window = np.hanning(len(n))
s1 = s1 * s1_window

s2 = A2 * np.sin(2 * np.pi * f2_center * n) * np.sin(np.pi * f2_bandwidth * n) / (np.pi * f2_bandwidth * n + 1e-10)
s2_window = np.hanning(len(n))
s2 = s2 * s2_window

x = s1 + s2

freqs_d = fftfreq(N_d, 1/fs)
S1_f = fft(s1)
S2_f = fft(s2)
X_f = fft(x)

fig3, axes3 = plt.subplots(3, 1, figsize=(12, 10))

axes3[0].stem(n[:50], s1[:50], 'b', linefmt='b-', markerfmt='bo', basefmt='k-')
axes3[0].set_xlabel('时间 (s)')
axes3[0].set_ylabel('幅度')
axes3[0].set_title(f'离散有用信号 s1(n) 时域波形 (fs={fs}Hz)')
axes3[0].grid(True, alpha=0.3)

axes3[1].stem(n[:50], s2[:50], 'r', linefmt='r-', markerfmt='ro', basefmt='k-')
axes3[1].set_xlabel('时间 (s)')
axes3[1].set_ylabel('幅度')
axes3[1].set_title(f'离散干扰信号 s2(n) 时域波形 (fs={fs}Hz)')
axes3[1].grid(True, alpha=0.3)

axes3[2].stem(n[:50], x[:50], 'g', linefmt='g-', markerfmt='go', basefmt='k-')
axes3[2].set_xlabel('时间 (s)')
axes3[2].set_ylabel('幅度')
axes3[2].set_title(f'离散合成信号 x(n) = s1(n) + s2(n) 时域波形 (fs={fs}Hz)')
axes3[2].grid(True, alpha=0.3)

plt.tight_layout()

fig4, axes4 = plt.subplots(3, 1, figsize=(12, 10))

s1_spectrum_d = np.abs(S1_f[:N_d//2])
freq_plot_d = freqs_d[:N_d//2]
axes4[0].plot(freq_plot_d, s1_spectrum_d, 'b', linewidth=1.5)
axes4[0].set_xlabel('频率 (Hz)')
axes4[0].set_ylabel('幅度')
axes4[0].set_title('离散有用信号 s1(n) 频谱 (FFT分析)')
axes4[0].grid(True, alpha=0.3)
axes4[0].set_xlim([0, 200])

s2_spectrum_d = np.abs(S2_f[:N_d//2])
axes4[1].plot(freq_plot_d, s2_spectrum_d, 'r', linewidth=1.5)
axes4[1].set_xlabel('频率 (Hz)')
axes4[1].set_ylabel('幅度')
axes4[1].set_title('离散干扰信号 s2(n) 频谱 (FFT分析)')
axes4[1].grid(True, alpha=0.3)
axes4[1].set_xlim([0, 200])

x_spectrum_d = np.abs(X_f[:N_d//2])
axes4[2].plot(freq_plot_d, x_spectrum_d, 'g', linewidth=1.5)
axes4[2].set_xlabel('频率 (Hz)')
axes4[2].set_ylabel('幅度')
axes4[2].set_title('离散合成信号 x(n) 频谱 (FFT分析)')
axes4[2].grid(True, alpha=0.3)
axes4[2].set_xlim([0, 200])

plt.tight_layout()
plt.show()

f_sample = fs
f_center_norm = f2_center / (f_sample/2)
f_bandwidth_norm = f2_bandwidth / (f_sample/2)

order = 6
rp = 1.0
rs = 45

f_stop_low = (f2_center - f2_bandwidth/2) / (f_sample/2)
f_stop_high = (f2_center + f2_bandwidth/2) / (f_sample/2)

b, a = signal.ellip(order, rp, rs, [f_stop_low, f_stop_high], btype='bandstop', fs=f_sample)

print(f"滤波器阶数：{order}")
print(f"滤波器类型：椭圆带阻滤波器")
print(f"通带纹波：{rp} dB")
print(f"阻带衰减：{rs} dB")
print(f"阻带频率范围：{f2_center - f2_bandwidth/2:.1f} Hz - {f2_center + f2_bandwidth/2:.1f} Hz")

w, h = signal.freqz(b, a, worN=8000, fs=f_sample)
magnitude = 20 * np.log10(np.abs(h) + 1e-10)
phase = np.angle(h)

idx_interference = np.argmin(np.abs(w - f2_center))
attenuation_at_interference = -magnitude[idx_interference]
print(f"\n滤波器验证：")
print(f"在干扰频率 {f2_center} Hz 处的衰减：{attenuation_at_interference:.2f} dB")

if attenuation_at_interference > 40:
    print("✓ 滤波器设计满足要求")
else:
    print("✗ 滤波器设计不满足要求")

fig5, (ax5_1, ax5_2) = plt.subplots(2, 1, figsize=(12, 8))

ax5_1.plot(w, magnitude, 'b', linewidth=2)
ax5_1.set_xlabel('频率 (Hz)')
ax5_1.set_ylabel('幅度 (dB)')
ax5_1.set_title('数字滤波器 H(z) 幅频特性')
ax5_1.grid(True, alpha=0.3)
ax5_1.set_xlim([0, 200])
ax5_1.set_ylim([-80, 5])

ax5_1.axvline(x=f1_center, color='g', linestyle='--', alpha=0.7, label=f'有用信号中心频率 ({f1_center} Hz)')
ax5_1.axvline(x=f2_center, color='r', linestyle='--', alpha=0.7, label=f'干扰信号中心频率 ({f2_center} Hz)')
ax5_1.axhline(y=-40, color='k', linestyle=':', alpha=0.5, label='-40 dB 衰减线')
ax5_1.fill_betweenx([-80, 5], f_stop_low*(f_sample/2), f_stop_high*(f_sample/2), 
                    alpha=0.2, color='red', label='阻带范围')
ax5_1.legend(loc='upper right')

ax5_2.plot(w, np.unwrap(phase), 'r', linewidth=2)
ax5_2.set_xlabel('频率 (Hz)')
ax5_2.set_ylabel('相位 (弧度)')
ax5_2.set_title('数字滤波器 H(z) 相频特性')
ax5_2.grid(True, alpha=0.3)
ax5_2.set_xlim([0, 200])

plt.tight_layout()
plt.show()


print(f"\n滤波器系数：")
print(f"分子系数 b (前馈系数): {b}")
print(f"分母系数 a (反馈系数): {a}")


y = signal.lfilter(b, a, x)

Y_f = fft(y)
y_spectrum = np.abs(Y_f[:N_d//2])

power_s1 = np.mean(s1**2)
power_s2 = np.mean(s2**2)
power_x = np.mean(x**2)
power_y = np.mean(y**2)

snr_before = 10 * np.log10(power_s1 / power_s2)
snr_after = 10 * np.log10(power_s1 / (power_y - power_s1))

print(f"\n信号功率分析：")
print(f"有用信号功率：{power_s1:.6f}")
print(f"干扰信号功率：{power_s2:.6f}")
print(f"输入信号功率：{power_x:.6f}")
print(f"输出信号功率：{power_y:.6f}")
print(f"滤波前信噪比：{snr_before:.2f} dB")
print(f"滤波后信噪比：{snr_after:.2f} dB")
print(f"信噪比改善：{snr_after - snr_before:.2f} dB")

fig6, axes6 = plt.subplots(2, 2, figsize=(14, 10))

axes6[0, 0].stem(n[:50], y[:50], 'b', linefmt='b-', markerfmt='bo', basefmt='k-')
axes6[0, 0].set_xlabel('时间 (s)')
axes6[0, 0].set_ylabel('幅度')
axes6[0, 0].set_title('滤波器输出 y(n) 时域波形')
axes6[0, 0].grid(True, alpha=0.3)

axes6[0, 1].plot(freq_plot_d, y_spectrum, 'b', linewidth=1.5)
axes6[0, 1].set_xlabel('频率 (Hz)')
axes6[0, 1].set_ylabel('幅度')
axes6[0, 1].set_title('滤波器输出 y(n) 频谱')
axes6[0, 1].grid(True, alpha=0.3)
axes6[0, 1].set_xlim([0, 200])

axes6[1, 0].plot(freq_plot_d, x_spectrum_d, 'r', linewidth=1.5, alpha=0.7, label='滤波前 x(n)')
axes6[1, 0].plot(freq_plot_d, y_spectrum, 'b', linewidth=1.5, label='滤波后 y(n)')
axes6[1, 0].set_xlabel('频率 (Hz)')
axes6[1, 0].set_ylabel('幅度')
axes6[1, 0].set_title('滤波前后频谱对比')
axes6[1, 0].grid(True, alpha=0.3)
axes6[1, 0].set_xlim([0, 200])
axes6[1, 0].legend()

axes6[1, 1].plot(freq_plot_d, s1_spectrum_d, 'g', linewidth=2, label='原始有用信号 s1(n)')
axes6[1, 1].plot(freq_plot_d, y_spectrum, 'b', linewidth=1.5, alpha=0.7, label='滤波器输出 y(n)')
axes6[1, 1].set_xlabel('频率 (Hz)')
axes6[1, 1].set_ylabel('幅度')
axes6[1, 1].set_title('有用信号与滤波器输出对比')
axes6[1, 1].grid(True, alpha=0.3)
axes6[1, 1].set_xlim([0, 100])
axes6[1, 1].legend()

plt.tight_layout()

fig7, axes7 = plt.subplots(3, 1, figsize=(12, 10))

axes7[0].plot(n[:100], s1[:100], 'g', linewidth=2, label='原始有用信号 s1(n)')
axes7[0].set_xlabel('时间 (s)')
axes7[0].set_ylabel('幅度')
axes7[0].set_title('原始有用信号')
axes7[0].grid(True, alpha=0.3)
axes7[0].legend()

axes7[1].plot(n[:100], x[:100], 'r', linewidth=1.5, alpha=0.7, label='滤波器输入 x(n)')
axes7[1].set_xlabel('时间 (s)')
axes7[1].set_ylabel('幅度')
axes7[1].set_title('滤波器输入信号（含干扰）')
axes7[1].grid(True, alpha=0.3)
axes7[1].legend()

axes7[2].plot(n[:100], y[:100], 'b', linewidth=1.5, label='滤波器输出 y(n)')
axes7[2].plot(n[:100], s1[:100], 'g--', linewidth=1, alpha=0.5, label='原始有用信号 (参考)')
axes7[2].set_xlabel('时间 (s)')
axes7[2].set_ylabel('幅度')
axes7[2].set_title('滤波器输出信号')
axes7[2].grid(True, alpha=0.3)
axes7[2].legend()

plt.tight_layout()
plt.show()