import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def exponential_sequence_fft_analysis():

    N = 64
    n = np.arange(N)
    alpha = 0.9
    x = alpha ** n

    X = np.fft.fft(x)
    freq = np.fft.fftfreq(N)

    x_recon = np.fft.ifft(X)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.stem(n, x, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
    plt.title('原始指数序列 x[n] = 0.9^n')
    plt.xlabel('时间序号 n')
    plt.ylabel('幅度')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 2)
    plt.stem(freq[:N // 2], np.abs(X)[:N // 2], linefmt='C1-', markerfmt='C1o', basefmt='C1-')
    plt.title('FFT幅频特性')
    plt.xlabel('归一化频率')
    plt.ylabel('|X[k]|')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 3)
    plt.stem(freq[:N // 2], np.angle(X)[:N // 2], linefmt='C2-', markerfmt='C2o', basefmt='C2-')
    plt.title('FFT相频特性')
    plt.xlabel('归一化频率')
    plt.ylabel('∠X[k] (rad)')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 4)
    plt.stem(n, np.real(x_recon), linefmt='C3-', markerfmt='C3o', basefmt='C3-')
    plt.title('IFFT重建序列')
    plt.xlabel('时间序号 n')
    plt.ylabel('幅度')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 5)
    error = np.abs(x - x_recon)
    plt.stem(n, error, linefmt='C4-', markerfmt='C4o', basefmt='C4-')
    plt.title('重建误差')
    plt.xlabel('时间序号 n')
    plt.ylabel('误差幅度')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 6, projection='polar')
    plt.plot(np.angle(X)[:N // 2], np.abs(X)[:N // 2], 'C5-', linewidth=1)
    plt.title('频谱极坐标表示')

    plt.tight_layout()
    plt.show()

    print("FFT/IFFT重建最大误差:", np.max(error))
    return x, X, x_recon

def system_response_fft_analysis():

    b = [1, 0.5]
    a = [1, -1.5, 0.7]

    N = 64
    n = np.arange(N)
    alpha = 0.8
    x = alpha ** n

    _, h = signal.dimpulse((b, a, 1), n=N)
    h = h[0].flatten()

    y_conv = np.convolve(x, h, mode='full')[:N]

    L = N + len(h) - 1
    X_fft = np.fft.fft(x, L)
    H_fft = np.fft.fft(h, L)
    Y_fft = X_fft * H_fft
    y_fft = np.fft.ifft(Y_fft)[:N]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.stem(n, x, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
    plt.title('输入序列 x[n]')
    plt.xlabel('时间序号 n')
    plt.ylabel('幅度')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.stem(n, h, linefmt='C1-', markerfmt='C1o', basefmt='C1-')
    plt.title('系统脉冲响应 h[n]')
    plt.xlabel('时间序号 n')
    plt.ylabel('幅度')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.stem(n, y_conv, linefmt='C2-', markerfmt='C2o', basefmt='C2-', label='时域卷积')
    plt.stem(n, np.real(y_fft), linefmt='C3--', markerfmt='C3x', basefmt='C3-', label='频域FFT')
    plt.title('系统输出响应比较')
    plt.xlabel('时间序号 n')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    error = np.abs(y_conv - np.real(y_fft))
    plt.stem(n, error, linefmt='C4-', markerfmt='C4o', basefmt='C4-')
    plt.title('时域与频域方法差异')
    plt.xlabel('时间序号 n')
    plt.ylabel('误差幅度')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("时域卷积与频域FFT方法最大差异:", np.max(error))
    return x, h, y_conv, y_fft

def fft_length_effect_analysis():

    b = [1, 0.5]
    a = [1, -1.5, 0.7]

    N = 32
    n = np.arange(N)
    x = 0.8 ** n

    _, h = signal.dimpulse((b, a, 1), n=N)
    h = h[0].flatten()

    fft_lengths = [32, 64, 128, 256]

    plt.figure(figsize=(12, 10))

    for i, L in enumerate(fft_lengths):

        X_fft = np.fft.fft(x, L)
        H_fft = np.fft.fft(h, L)
        Y_fft = X_fft * H_fft
        y_fft = np.fft.ifft(Y_fft)[:N]

        y_ref = np.convolve(x, h, mode='full')[:N]

        plt.subplot(2, 2, i + 1)
        plt.stem(n, y_ref, linefmt='C0-', markerfmt='C0o', basefmt='C0-', label='时域参考')
        plt.stem(n, np.real(y_fft), linefmt='C1--', markerfmt='C1x', basefmt='C1-', label=f'FFT长度={L}')
        plt.title(f'FFT长度 = {L}')
        plt.xlabel('时间序号 n')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)

        error = np.max(np.abs(y_ref - np.real(y_fft)))
        plt.text(0.05, 0.95, f'最大误差: {error:.2e}',
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()

    return fft_lengths


def continuous_signal_analysis():

    alpha = 0.9


    T_obs = 5.0


    fs_list = [10, 20, 50, 100]

    plt.figure(figsize=(12, 10))

    for i, fs in enumerate(fs_list):

        Ts = 1.0 / fs
        N = int(T_obs * fs)

        t_continuous = np.linspace(0, T_obs, 1000)
        t_sampled = np.arange(0, T_obs, Ts)

        x_continuous = np.exp(-alpha * t_continuous)
        x_sampled = np.exp(-alpha * t_sampled)

        X_fft = np.fft.fft(x_sampled)
        freq = np.fft.fftfreq(len(x_sampled), Ts)

        plt.subplot(2, 2, i + 1)

        plt.plot(t_continuous, x_continuous, 'C0-', linewidth=1, alpha=0.7, label='连续信号')
        plt.stem(t_sampled, x_sampled, linefmt='C1-', markerfmt='C1o', basefmt='C1-', label='采样点')
        plt.title(f'采样频率 fs = {fs} Hz')
        plt.xlabel('时间 t (秒)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fs_list

def observation_time_fft_length_effect():

    alpha = 0.9

    fs = 50
    Ts = 1.0 / fs

    T_obs_list = [1.0, 2.0, 5.0, 10.0]

    plt.figure(figsize=(12, 10))

    for i, T_obs in enumerate(T_obs_list):

        N_original = int(T_obs * fs)
        t_sampled = np.arange(0, T_obs, Ts)
        x_sampled = np.exp(-alpha * t_sampled)

        fft_lengths = [N_original, 2 * N_original, 4 * N_original]

        plt.subplot(2, 2, i + 1)

        for L in fft_lengths:

            X_fft = np.fft.fft(x_sampled, L)
            freq = np.fft.fftfreq(L, Ts)

            positive_freq = freq[:L // 2]
            positive_spectrum = np.abs(X_fft)[:L // 2]

            plt.plot(positive_freq, positive_spectrum,
                     label=f'FFT长度={L}', linewidth=2)

        f_theory = np.linspace(0, fs / 2, 1000)
        X_theory = 1.0 / np.sqrt(alpha ** 2 + (2 * np.pi * f_theory) ** 2)
        plt.plot(f_theory, X_theory, 'k--', linewidth=2, label='理论频谱')

        plt.title(f'观测时间 T = {T_obs} 秒')
        plt.xlabel('频率 f (Hz)')
        plt.ylabel('幅度谱')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 10)

    plt.tight_layout()
    plt.show()

    return T_obs_list

def main():

    print("=" * 60)
    print("实验四: 信号的频谱分析")
    print("=" * 60)

    # 1. 指数序列的FFT和IFFT分析
    print("\n1. 指数序列的FFT和IFFT分析")
    x, X, x_recon = exponential_sequence_fft_analysis()

    # 2. 利用FFT计算系统输出响应
    print("\n2. 利用FFT计算系统输出响应")
    x_input, h_system, y_conv, y_fft = system_response_fft_analysis()

    # 3. FFT计算长度对系统输出的影响
    print("\n3. FFT计算长度对系统输出的影响分析")
    fft_lengths = fft_length_effect_analysis()

    # 4. 连续时间信号的采样和频谱分析
    print("\n4. 连续时间信号的采样和频谱分析")
    fs_list = continuous_signal_analysis()

    # 5. 观测时间和FFT长度对频谱的影响
    print("\n5. 观测时间和FFT长度对频谱的影响分析")
    T_obs_list = observation_time_fft_length_effect()




if __name__ == "__main__":
    main()