%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 混频器模型：只考虑相位噪声
% 输出 = 输入信号 × 带有相位噪声的本振
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear; clc;

%% 1. 仿真参数
fs      = 10e6;          % 采样率 (Hz)
Tsim    = 1e-3;          % 仿真时长 (s)
N       = floor(Tsim*fs);% 采样点数
t       = (0:N-1)/fs;    % 时间向量

%% 2. 输入信号 (单音 RF 信号)
% 本示例为展示频域搬移，使用复混频方式，直接将频谱搬移显示。
% 我们改用等效方式：输入为基带/中频信号，混频后用低通滤波观察。
% 为简单起见，假设我们只关心混频后下变频信号，采用复混频减少采样率要求。

% 重新设计为复基带仿真：
% 输入信号为复单音，频率 f_IF_in，本振有相位噪声，输出下变频到基带附近。
f_IF_in = 1e6;          % 输入中频 (Hz)，例如 1 MHz
x_baseband = exp(1j*2*pi*f_IF_in*t);  % 复信号，幅度 1

%% 3. 定义相位噪声模板 (单边带 dBc/Hz)
f_offset = [1e3, 10e3, 100e3, 1e6];   % 频偏 (Hz)
L_dBc    = [-60,  -80,  -110,  -130]; % 相位噪声 (dBc/Hz)

% 注意：模板需覆盖 0~fs/2，且低频段不能太低，避免积分发散。
% 此处设置的相位噪声较为平缓，适合仿真。

%% 4. 由相位噪声模板生成时域相位噪声序列 phi_n[n]
% 步骤：
%   a) 将 dBc/Hz 转换为双边带功率谱 S_phi (rad^2/Hz)
%   b) 设计 FIR 滤波器，使其幅频响应为 sqrt(S_phi)
%   c) 对高斯白噪声滤波，得到 phi_n[n]

% 4.1 扩展模板到全部正频率（插值）
% 构建线性频率轴上的目标功率谱
freq_axis = linspace(0, fs/2, N/2+1);  % 单边频率向量
% 插值得到目标 S_phi 值 (双边带)
L_interp = interp1(f_offset, L_dBc, freq_axis, 'linear', 'extrap');
% 限制低频最小频偏，避免无穷大（本振近端相位噪声通常平缓或截断）
L_interp(freq_axis < min(f_offset)) = L_dBc(1); 
S_phi_target = 2 * 10.^(L_interp/10);  % 双边带谱 rad^2/Hz

% 注意：freq_axis(1) = 0，相位噪声在 0 频偏处理论上无定义，设为有限值
S_phi_target(1) = S_phi_target(2);  % 用最近值填充

% 4.2 用频域法生成有色噪声（直接频谱整形）
% 生成白噪声的 FFT
white_noise = randn(1, N);
W = fft(white_noise);

% 构造正频率部分的滤波器幅频响应（注意双边谱对称）
% 期望的 phi_n 双边谱为 S_phi_target，均方根为 sqrt(S_phi_target*fs/N)？
% 利用关系：若 phi_n 由白噪声通过滤波器 H(f) 得到，则 S_phi(f) = |H(f)|^2 * sigma^2/fs
% 若白噪声方差为 1，则 |H(f)| = sqrt(S_phi(f) * fs)
% 我们在频域直接乘以滤波因子。
H_single = sqrt(S_phi_target * fs);  % 单边滤波器幅度
% 构造双边谱（共轭对称）
H_full = [H_single, H_single(end-1:-1:2)]; % 保证长度为 N
% 频域相乘
Phi_n_fft = W .* H_full;
phi_n = real(ifft(Phi_n_fft));   % 时域相位噪声，单位为 rad

% 归一化（可选）：确保相位噪声均方根值在合理范围
phi_n = phi_n - mean(phi_n);     % 去直流

%% 5. 生成本振信号（复本振，便于观察相位噪声效应）
f_LO = 0;  % 设定本振频率，若用复基带表示，设 f_LO=0 可模拟零中频下变频
% 实际混频器模型：输出 = x * exp(-1j*(2*pi*f_LO*t + phi_n))
% 我们模拟零中频：本振为 exp(-1j*phi_n(t))
LO = exp(-1j * phi_n);    % 只含相位噪声

% 混频输出 (下变频后的复基带信号)
y = x_baseband .* LO;

%% 6. 频谱分析
win = blackmanharris(N)';
win_gain = mean(win);
yw = y .* win;
Y = fft(yw) / (N * win_gain);
Y_dB = 20*log10(abs(fftshift(Y)) + eps);
f_axis = (-N/2:N/2-1) * fs / N;  % 双边频率

% 理想情况 (无相位噪声) 的混频输出对比
y_ideal = x_baseband .* exp(-1j*0);  % 本振无噪声
yw_ideal = y_ideal .* win;
Y_ideal = fft(yw_ideal) / (N * win_gain);
Y_ideal_dB = 20*log10(abs(fftshift(Y_ideal)) + eps);

%% 7. 绘图
figure('Position', [100, 100, 1000, 500]);
plot(f_axis/1e3, Y_dB, 'b-', 'LineWidth', 1); hold on;
plot(f_axis/1e3, Y_ideal_dB, 'r--', 'LineWidth', 1.5);
xlabel('Frequency (kHz)'); ylabel('Magnitude (dB)');
title('Mixer Output Spectrum (with Phase Noise)');
legend('With Phase Noise', 'Ideal (No Phase Noise)');
grid on; xlim([-2*f_IF_in/1e3, 2*f_IF_in/1e3]); % 观察信号附近

%% 8. 打印信息
fprintf('混频器相位噪声模型仿真完成。\n');
fprintf('输入中频: %.1f kHz\n', f_IF_in/1e3);
fprintf('相位噪声模板: \n');
for k=1:length(f_offset)
    fprintf('  @ %6.0f Hz : %d dBc/Hz\n', f_offset(k), L_dBc(k));
end