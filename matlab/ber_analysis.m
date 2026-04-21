%% ============================================================
%  DVB-S2X 误码率分析工具
%  使用最新MATLAB函数: ldpcEncode, ldpcDecode, dvbsLDPCPCM,
%                       pskmod, qammod, qamdemod, pskdemod
%  调用格式:
%    ber_analysis_main('PSK',  4,   '1/2', 64800)
%    ber_analysis_main('PSK',  8,   '3/4', 64800)
%    ber_analysis_main('APSK', 16,  '3/4', 64800)
%    ber_analysis_main('QAM',  64,  '2/3', 16200)
%    ber_analysis_main('QAM',  256, '5/6', 64800)
%% ============================================================

function ber_analysis_main(mod_type, mod_order, code_rate_str, frame_length)

%% ---------- 输入检查 ----------
validate_inputs(mod_type, mod_order, code_rate_str, frame_length);
mod_type = upper(mod_type);
code_rate = parse_cr(code_rate_str);
bits_per_sym = log2(mod_order);

%% ---------- LDPC 校验矩阵 + 配置对象 ----------
fprintf('\n正在加载 DVB-S2 LDPC 校验矩阵...\n');
if frame_length == 64800,     frame_type = 'normal';
elseif frame_length == 32400, frame_type = 'medium';
else,                         frame_type = 'short';
end


pcm          = dvbsLDPCPCM(code_rate_str, frame_type);
enc_ldpc_cfg = ldpcEncoderConfig(pcm);
dec_ldpc_cfg = ldpcDecoderConfig(pcm);
n_ldpc       = enc_ldpc_cfg.BlockLength;
k_ldpc       = enc_ldpc_cfg.NumInformationBits;  % BCH输出必须等于此值
use_ldpc     = true;
fprintf('✓ LDPC (%d, %d)  校验矩阵已加载\n', n_ldpc, k_ldpc);


%% ---------- 编码链路说明 ----------
% DVB-S2 标准链路: Kbch bits → BCH → Nbch=k_ldpc bits → LDPC → n_ldpc bits
% comm.BCHEncoder 对DVB-S2缩短码支持不稳定，此处仅做LDPC仿真
% LDPC已提供绝大部分编码增益，BCH仅在BER<1e-7时才有微弱增益
use_bch = false;
k_info  = k_ldpc;   % 信息比特数 = LDPC输入位数
fprintf('  编码: 仅LDPC  (信息位 %d → 码字 %d)\n', k_info, n_ldpc);

%% ---------- SNR 范围 ----------
spec_eff = bits_per_sym * code_rate;
shannon  = 10*log10((2^spec_eff - 1) / spec_eff);
% 瀑布区前后用细步长0.25dB，远离瀑布区用粗步长1dB
EbN0_dB = unique([ ...
    (shannon - 2)   : 1.0  : (shannon + 0.5), ...  % 瀑布前粗扫
    (shannon + 0.5) : 0.2 : (shannon + 3.0), ...  % 瀑布区细扫
    (shannon + 3.0) : 0.1  : (shannon + 10.0)  ...  % 瀑布后粗扫
]);
EbN0_dB = max(EbN0_dB, -3);
EsN0_dB = EbN0_dB + 10*log10(spec_eff);

%% ---------- 仿真参数 ----------
num_frames      = 100;   % 每SNR点最大帧数
min_err_bits    = 30;   % 累积此数量错误比特后提前停止（高BER区）
min_frames_low  = 30;    % 高BER区最少帧数
consec_zero_max = 3;     % 连续 N 个SNR点BER=0后停止仿真
ldpc_max_iter   = 50;

%% ---------- 打印配置 ----------
fprintf('\n=========================================\n');
fprintf('  DVB-S2X BER 仿真\n');
fprintf('=========================================\n');
fprintf('  调制: %s-%d\n', mod_type, mod_order);
fprintf('  码率: %s  频谱效率: %.3f bit/s/Hz\n', code_rate_str, spec_eff);
fprintf('  码长: %d  香农限: %.2f dB\n', frame_length, shannon);
fprintf('  信息比特/帧: %d\n', k_info);
fprintf('=========================================\n\n');

%% ---------- 预分配结果 ----------
N = length(EbN0_dB);
ber_coded   = zeros(1, N);
consec_zero = 0;

%% ============================================================
%  主仿真循环
%% ============================================================
for idx = 1:N
    EsN0_lin = 10^(EsN0_dB(idx)/10);
    sigma    = sqrt(1 / (2 * EsN0_lin));

    err_bits  = 0;
    tot_bits  = 0;
    tot_frm   = 0;

    for frm = 1:num_frames
        %% ---- 信息比特 ----
        info = logical(randi([0 1], k_info, 1));

        %% ---- LDPC 编码 ----
        if use_ldpc
            tx_bits = ldpcEncode(info, enc_ldpc_cfg);
        else
            tx_bits = info;
        end

        %% ---- 调制 ----
        tx_syms = do_modulate(tx_bits, mod_type, mod_order, code_rate_str);

        %% ---- AWGN ----
        rx_syms = tx_syms + sigma*(randn(size(tx_syms)) + 1j*randn(size(tx_syms)));

        %% ---- 软解调 → LLR ----
        llr = do_demod_llr(rx_syms, mod_type, mod_order, sigma, code_rate_str);

        %% ---- LDPC 解码 ----
        % OutputFormat='info': 直接输出 k_ldpc 位信息比特，无需手动截取
        if use_ldpc
            rx_info = logical(ldpcDecode(llr, dec_ldpc_cfg, ldpc_max_iter, ...
                                  'DecisionType', 'hard', ...
                                  'OutputFormat', 'info'));
        else
            rx_info = llr < 0;
        end

        %% ---- 统计 ----
        n_cmp    = min(k_info, length(rx_info));
        bit_err  = sum(info(1:n_cmp) ~= rx_info(1:n_cmp));

        err_bits = err_bits + bit_err;
        tot_bits = tot_bits + n_cmp;
        tot_frm  = tot_frm  + 1;

        % 高BER区：累积足够错误后提前结束本SNR点
        if err_bits >= min_err_bits && frm >= min_frames_low, break; end
    end

    ber_coded(idx) = err_bits / max(tot_bits, 1);

    fprintf('Eb/N0=%5.1f dB  BER=%.2e  帧=%d\n', ...
        EbN0_dB(idx), ber_coded(idx), tot_frm);

    % 瀑布后连续BER=0则停止，避免浪费时间
    if ber_coded(idx) == 0
        consec_zero = consec_zero + 1;
        if consec_zero >= consec_zero_max
            EbN0_dB   = EbN0_dB(1:idx);
            ber_coded = ber_coded(1:idx);
            fprintf('  → 连续 %d 个SNR点 BER=0，提前终止仿真\n', consec_zero_max);
            break;
        end
    else
        consec_zero = 0;
    end
end

%% ---------- 绘图 ----------
plot_ber(EbN0_dB, ber_coded, mod_type, mod_order, code_rate_str, frame_length, spec_eff, shannon);

end % ===== 主函数结束 =====


%% ============================================================
%%  子函数
%% ============================================================

% ---- 输入验证 ----
function validate_inputs(mt, mo, cr, fl)
    assert(ismember(upper(mt), {'PSK','APSK','QAM'}), '调制方式: PSK/APSK/QAM');
    vo.PSK=[4,8]; vo.APSK=[8,16,32,64,128,256]; vo.QAM=[16,64,256];
    assert(ismember(mo, vo.(upper(mt))), '调制阶数不支持');
    % assert(ismember(cr, {'1/2','3/5','2/3','3/4','4/5','5/6','8/9','9/10'}), '码率不支持');
    assert(ismember(fl, [16200,32400,64800]), '码长: 16200/32400/64800');
end

% ---- 码率解析 ----
function r = parse_cr(s)
    p = strsplit(s,'/'); r = str2double(p{1})/str2double(p{2});
end

% ---- 调制 ----
function syms = do_modulate(bits, mod_type, mod_order, cr_str)
    bits = double(bits(:));
    bps  = log2(mod_order);
    pad  = mod(length(bits), bps);
    if pad > 0, bits = [bits; zeros(bps-pad, 1)]; end

    switch mod_type
        case 'PSK'
            % DVB-S2 标准相位偏转 π/M，Gray 映射，直接比特输入
            phase_off = pi/mod_order;
            syms = pskmod(bits, mod_order, phase_off, 'gray', ...
                          'InputType', 'bit');

        case 'APSK'
            syms = dvbsapskmod(bits, mod_order, 's2', cr_str, ...
                               'InputType', 'bit', ...
                               'UnitAveragePower', true);

        case 'QAM'
            syms = qammod(bits, mod_order, 'gray', ...
                          'InputType', 'bit', ...
                          'UnitAveragePower', true);
    end
    % 不做二次归一化：pskmod/qammod/dvbsapskmod(UnitAveragePower=true)
    % 输出已是单位平均功率，再归一化会破坏解调端的 NoiseVariance 对应关系
end

% ---- 软解调 → LLR ----
% NoiseVariance = 每符号噪声功率 = 2*sigma^2 (sigma为每IQ分量标准差)
% pskdemod/qamdemod/dvbsapskdemod 的 NoiseVariance 均指每符号总功率
function llr = do_demod_llr(rx, mod_type, mod_order, sigma, cr_str)
    noiseVar = 2 * sigma^2;   % = N0 (归一化符号能量Es=1时)

    switch mod_type
        case 'PSK'
            phase_off = pi/mod_order;
            llr = pskdemod(rx, mod_order, phase_off, 'gray', ...
                           'OutputType', 'approxllr', 'NoiseVariance', noiseVar);

        case 'APSK'
            llr = dvbsapskdemod(rx, mod_order, 's2', cr_str, ...
                                'OutputType', 'approxllr', ...
                                'NoiseVariance', noiseVar, ...
                                'UnitAveragePower', true);

        case 'QAM'
            llr = qamdemod(rx, mod_order, 'gray', ...
                           'OutputType', 'approxllr', ...
                           'UnitAveragePower', true, ...
                           'NoiseVariance', noiseVar);
    end
    llr = double(llr(:));
    llr = max(min(llr, 100), -100);
end

% ---- 理论 BER (未编码) ----
function ber = theoretical_ber(mod_type, M, EbN0_dB)
    EbN0 = 10^(EbN0_dB/10);
    switch mod_type
        case 'PSK'
            k = log2(M);
            if M == 4
                ber = erfc(sqrt(EbN0)) / 2;
            else
                ber = (2/k) * qfunc(sqrt(2*k*EbN0)*sin(pi/M));
            end
        case {'APSK','QAM'}
            k = log2(M);
            if sqrt(M)==floor(sqrt(M))
                ber = (4/k)*(1-1/sqrt(M))*qfunc(sqrt(3*k*EbN0/(M-1)));
            else
                ber = (4/k)*0.8*qfunc(sqrt(2*k*EbN0/M));
            end
    end
    ber = max(ber, 1e-10);
end

% ---- 绘图 ----
function plot_ber(EbN0_dB, ber_c, mod_type, mod_order, cr_str, flen, spec_eff, shannon)

    BG   = [0.10 0.10 0.13];
    CARD = [0.13 0.13 0.17];
    CT   = [0.90 0.90 0.92];
    CG   = [0.24 0.24 0.30];
    C1   = [0.22 0.82 0.52];   % 绿  编码BER
    C3   = [0.65 0.65 0.80];   % 灰蓝 未编码理论
    C4   = [0.95 0.82 0.22];   % 黄  香农限
    C5   = [0.55 0.88 0.98];   % 青  BER=1e-6

    fig = figure('Name','DVB-S2X BER Analysis', ...
        'Position',[80 60 1120 660], 'Color',BG);

    ax = axes('Parent',fig, 'Position',[0.07 0.10 0.60 0.84], ...
        'Color',CARD, 'XColor',CT, 'YColor',CT, ...
        'YScale','log', ...                          % 强制对数纵坐标
        'GridColor',CG, 'GridAlpha',0.55, ...
        'MinorGridColor',CG, 'MinorGridAlpha',0.25, ...
        'XGrid','on','YGrid','on','MinorGridLineStyle',':', ...
        'FontSize',10);

    hold(ax,'on');

    %% 理论BER：SNR范围延伸到足够高，让曲线有斜度可见
    snr_theory = linspace(EbN0_dB(1)-1, EbN0_dB(end)+3, 200);
    ber_theory = arrayfun(@(s) theoretical_ber(mod_type, mod_order, s), snr_theory);
    semilogy(ax, snr_theory, ber_theory, '--', ...
        'Color',C3,'LineWidth',1.5,'DisplayName', ...
        sprintf('未编码 BER（%s-%d 理论）', mod_type, mod_order));

    %% 编码后BER（仿真值）——只画有效点
    v = ber_c > 0;
    if sum(v) >= 2
        semilogy(ax, EbN0_dB(v), ber_c(v), '-o', ...
            'Color',C1,'LineWidth',2,'MarkerSize',7, ...
            'MarkerFaceColor',C1, ...
            'DisplayName',sprintf('编码 BER  %s-%d  R=%s  (LDPC)', ...
            mod_type, mod_order, cr_str));
    end

    %% 香农限竖线
    xline(ax, shannon, '--', 'Color',C4,'LineWidth',1.8, ...
        'DisplayName',sprintf('香农限 %.2f dB', shannon));

    %% BER 参考横线
    yline(ax, 1e-6, ':', 'Color',C5,'LineWidth',1.3, ...
        'DisplayName','BER = 10^{-6}');
    yline(ax, 1e-4, ':', 'Color',[0.85 0.72 0.28],'LineWidth',1.0, ...
        'DisplayName','BER = 10^{-4}');

    hold(ax,'off');

    %% 纵坐标：对数，下限取仿真有效值最小一个数量级，上限1
    valid_ber = ber_c(ber_c > 0);
    if ~isempty(valid_ber)
        y_floor = 10^(floor(log10(min(valid_ber))) - 1);
        y_floor = min(y_floor, 1e-7);   % 至少显示到 1e-7
    else
        y_floor = 1e-7;
    end
    ylim(ax, [y_floor, 1]);
    xlim(ax, [min(snr_theory(1), EbN0_dB(1)-0.5),  EbN0_dB(end)+0.5]);

    xlabel(ax,'E_b/N_0  (dB)','FontSize',12,'FontWeight','bold','Color',CT);
    ylabel(ax,'误码率 BER（对数坐标）','FontSize',12,'FontWeight','bold','Color',CT);
    title(ax, sprintf('DVB-S2X 误码率  |  %s-%d  |  码率 %s  |  帧长 %d  |  效率 %.2f bit/s/Hz', ...
        mod_type, mod_order, cr_str, flen, spec_eff), ...
        'FontSize',12,'FontWeight','bold','Color',CT);
    legend(ax,'Location','southwest','FontSize',9.5, ...
        'TextColor',CT,'Color',[0.15 0.15 0.19],'EdgeColor',CG);

    %% 右侧参数面板
    ax2 = axes('Parent',fig,'Position',[0.71 0.10 0.27 0.84]);
    axis(ax2,'off'); set(ax2,'Color',BG);

    % 编码增益：在 BER=1e-4 处，编码曲线与未编码理论曲线的SNR差值
    tgt = 1e-4;
    coding_gain = 0;
    try
        vc = ber_c > 0 & ~isnan(ber_c);
        if sum(vc) >= 2
            s_c = interp1(log10(ber_c(vc)+eps), EbN0_dB(vc), log10(tgt), 'linear','extrap');
            s_u = interp1(log10(ber_theory+eps), snr_theory,  log10(tgt), 'linear','extrap');
            coding_gain = max(0, min(s_u - s_c, 15));
        end
    catch; end

    r = parse_cr(cr_str);

    lines = {
        '┌─ 系统配置 ────────────────┐'
        sprintf('  调制方式  :  %s-%d', mod_type, mod_order)
        sprintf('  码率      :  %s  (%.4f)', cr_str, r)
        sprintf('  帧长      :  %d bits', flen)
        sprintf('  频谱效率  :  %.3f bit/s/Hz', spec_eff)
        '├─ 编码参数 ────────────────┤'
        '  编码      :  LDPC (DVB-S2)'
        sprintf('  码长      :  %d bits', flen)
        '  LDPC迭代  :  ≤ 50 次'
        '├─ 仿真函数 ────────────────┤'
        '  dvbsLDPCPCM'
        '  ldpcEncode / ldpcDecode'
        '  pskmod / pskdemod'
        '  qammod / qamdemod'
        '├─ 性能估算 ────────────────┤'
        sprintf('  香农限    :  %.2f dB', shannon)
        sprintf('  编码增益  :  ≈ %.1f dB', coding_gain)
        '└──────────────────────────┘'
    };
    for k = 1:length(lines)
        text(ax2, 0.03, 1.02-(k-1)*0.055, lines{k}, ...
            'Units','normalized','Color',CT,'FontSize',9.0, ...
            'FontName','Courier New');
    end

    %% ------ 保存 ------
    fname = sprintf('BER_%s%d_R%s_N%d.png', ...
        mod_type, mod_order, strrep(cr_str,'/','-'), flen);
    try
        exportgraphics(fig, fname, 'Resolution',150, ...
            'BackgroundColor',BG);
    catch
        saveas(fig, fname);
    end
    fprintf('\n图像已保存: %s\n仿真完成！\n', fname);
end