function [ber, qef] = compute_coded_ber(mod_str, rate_str, fecframe, esno_db)
% COMPUTE_CODED_BER  DVB-S2/S2X LDPC+BCH 编码后 BER 瀑布曲线近似
%
% mod_str  : 调制方式
% rate_str : 码率字符串，如 '3/4'
% fecframe : 码帧长度，16200 / 32400 / 64800
% esno_db  : Es/No 数组 (dB)
%
% 返回:
%   ber : 编码后 BER 行向量；若无门限定义则返回空 []
%   qef : QEF 门限 Es/No (dB)；无定义时返回 NaN

qef = get_qef_threshold(mod_str, rate_str);
if isnan(qef)
    ber = [];
    return;
end

% 码长 → 瀑布斜率（码越长越陡）
switch fecframe
    case 16200,  k = 2.5;
    case 32400,  k = 3.2;
    case 64800,  k = 4.0;
    otherwise,   k = 3.0;
end

ber_floor = 1e-8;   % error floor（保守估计，实测约 1e-9~1e-10）

% 瀑布顶部（门限以下 3dB 处的未编码 BER）
ber_top = min(compute_uncoded_ber(mod_str, qef - 3.0), 0.3);

esno_db = esno_db(:)';          % 行向量
ber     = zeros(size(esno_db));

for i = 1:numel(esno_db)
    x = esno_db(i);
    if x < qef - 5
        % 远低于门限：近似为未编码 BER
        ber(i) = min(compute_uncoded_ber(mod_str, x), 0.5);
    elseif x > qef + 3
        % 远高于门限：到达 QEF，取 error floor
        ber(i) = ber_floor;
    else
        % 瀑布段：Sigmoid 近似
        sig    = 1.0 / (1.0 + exp(k * (x - qef)));
        ber(i) = max(ber_floor, ber_top * sig + ber_floor * (1 - sig));
    end
end

ber = min(max(ber, 1e-12), 0.5);
end