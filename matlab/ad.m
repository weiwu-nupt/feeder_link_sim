%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADC 模型：只考虑量化位数和杂散
% 模型输出 = 量化( 输入信号 + 杂散信号 )
% 忽略热噪声、时钟抖动、INL/DNL 等非理想因素
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear; clc;

%% 1. 基本参数设置
N      = 12;            % 量化位数 (bit)
fs     = 1e6;           % 采样频率 (Hz)
Nfft   = 65536;         % FFT 点数 (建议使用 2 的幂)
t      = (0:Nfft-1)/fs; % 时间向量

% 归一化满量程：输入/输出幅度在 [-1, 1] 之间
% 满量程正弦波幅度为 1，对应 0 dBFS
q = 2 / 2^N;            % 量化步长 (归一化后)

%% 2. 输入信号 (正弦波)
fin = 100e3;            % 输入频率 (Hz)，最好不与 fs/Nfft 成整数倍关系
Ain = 0.9;              % 归一化幅度 (<1)，对应约 -0.915 dBFS
x   = Ain * sin(2*pi*fin*t);

%% 3. 杂散信号 (多个离散频率正弦波叠加)
f_spur = [50e3, 200e3, 350e3];          % 杂散频率 (Hz)，必须 < fs/2
A_dBc  = [-80,   -85,    -90];          % 相对于满量程幅度 (1) 的杂散电平 (dBc)
A_spur = Ain * 10.^(A_dBc/20);               % 线性杂散幅度

s = zeros(1, Nfft);
for k = 1:length(f_spur)
    s = s + A_spur(k) * sin(2*pi*f_spur(k)*t);
end

%% 4. 量化过程 (中平型量化器，二进制补码输出)
vin = x + s;                              % ADC 输入 (含杂散)
code = round(vin / q);                    % 理想量化，无失调

% 限幅处理：防止超量程，输出码值范围 [-2^(N-1), 2^(N-1)-1]
code = max(min(code, 2^(N-1)-1), -2^(N-1));
y = code * q;                             % 重构的归一化输出 (供分析用)

%% 5. 频谱分析
win = blackmanharris(Nfft)';              % 窗函数 (改善频谱泄漏)
win_gain = mean(win);                     % 相干增益
yw = y .* win;
Y = fft(yw) / (Nfft * win_gain);         % 归一化 FFT，使得峰值幅度正确

% 取正频率部分 (含直流)
Y_single = Y(1:Nfft/2+1);
Y_single(2:end-1) = 2 * Y_single(2:end-1); % 单边谱幅度 (峰值)
f_axis = (0:Nfft/2) * fs / Nfft;

% 转换为 信号幅度电平dBFS (满量程正弦波幅度 1 对应 0 dBFS)
Y_dB = 20*log10(abs(Y_single) + eps);

%% 6. 查找信号、杂散和 SFDR
[~, idx_sig] = min(abs(f_axis - fin));
sig_dB = Y_dB(idx_sig);

% 排除直流、信号及其附近泄漏区域，寻找最大杂散
exclude_bins = 10;  % 信号左右排除的 bin 数
idx_exclude = [1, max(1, idx_sig-exclude_bins):min(length(f_axis), idx_sig+exclude_bins)];
idx_valid = setdiff(1:length(f_axis), idx_exclude);

if ~isempty(idx_valid)
    [spur_max_dB, max_idx] = max(Y_dB(idx_valid));
    sfdr = sig_dB - spur_max_dB;   % SFDR = 信号功率(dBFS) - 最大杂散功率(dBFS)
    spur_freq = f_axis(idx_valid(max_idx));
else
    spur_max_dB = -inf;
    sfdr = inf;
    spur_freq = NaN;
end

%% 7. 估计量化噪声基底 (取频谱中远离信号和杂散的 bin 平均)
noise_exclude = 50;  % 信号左右排除的 bin
idx_noise_excl = [1, max(1,idx_sig-noise_exclude):min(length(f_axis),idx_sig+noise_exclude)];
for k = 1:length(f_spur)  % 也排除杂散频率附近
    [~, idx_spur] = min(abs(f_axis - f_spur(k)));
    idx_noise_excl = [idx_noise_excl, ...
        max(1,idx_spur-5):min(length(f_axis),idx_spur+5)];
end
idx_noise = setdiff(2:length(f_axis), unique(idx_noise_excl));
if ~isempty(idx_noise)
    noise_pow = mean( abs(Y_single(idx_noise)).^2 );
    noise_floor_dB = 10*log10(noise_pow + eps);
else
    noise_floor_dB = -inf;
end

%% 8. 绘图
figure('Position', [100, 100, 900, 500]);
plot(f_axis/1e3, Y_dB, 'b-', 'LineWidth', 0.5); hold on;
plot(fin/1e3, sig_dB, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
if ~isempty(idx_valid)
    plot(spur_freq/1e3, spur_max_dB, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
end
if ~isempty(idx_noise)
    yline(noise_floor_dB, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5);
end

xlabel('频率 (kHz)');
ylabel('幅度 (dBFS)');
title(sprintf(['ADC 模型: N = %d 位, SFDR = %.1f dBc, ' ...
    '噪声底 ≈ %.1f dBFS'], N, sfdr, noise_floor_dB));
legend({'频谱', '信号', '最大杂散', '平均噪声底'}, 'Location', 'best');
grid on; xlim([0, fs/2e3]);
hold off;

%% 9. 打印关键结果
fprintf('================ ADC 模型结果 ================\n');
fprintf('量化位数 N = %d bits\n', N);
fprintf('量化步长 (归一化) = %.6e\n', q);
fprintf('信号幅度: %.2f (%.2f dBFS)\n', Ain, 20*log10(Ain));
fprintf('杂散数量: %d\n', length(f_spur));
fprintf('SFDR = %.2f dBc (最大杂散在 %.1f kHz, %.2f dBFS)\n', ...
    sfdr, spur_freq/1e3, spur_max_dB);
fprintf('估计量化噪声基底 = %.2f dBFS\n', noise_floor_dB);
fprintf('================================================\n');