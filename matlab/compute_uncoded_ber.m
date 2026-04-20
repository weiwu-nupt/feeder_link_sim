function ber = compute_uncoded_ber(mod_str, esno_db)
% COMPUTE_UNCODED_BER  未编码 BER 解析式（AWGN，Grey 编码）
%
% mod_str  : 调制方式字符串，如 'QPSK' '8PSK' '16APSK' '64QAM'
% esno_db  : Es/No 数组 (dB)
% 返回 ber : 行向量，与 esno_db 等长

esno_lin = 10.^(esno_db(:)' / 10);   % 统一转为行向量

if strcmpi(mod_str, 'QPSK')
    ber = qfunc(sqrt(2 * esno_lin));

elseif strcmpi(mod_str, '8PSK')
    m   = 3;   % log2(8)
    ber = (1/m) * erfc(sqrt(esno_lin * (sin(pi/8))^2 * m));

elseif ~isempty(regexpi(mod_str, 'APSK$'))
    M   = get_mod_order(mod_str);
    ber = apsk_ber_approx(esno_lin, M);

elseif ~isempty(regexpi(mod_str, 'QAM$'))
    M   = get_mod_order(mod_str);
    m   = log2(M);
    ber = (4/m) * (1 - 1/sqrt(M)) * qfunc(sqrt(3 * m * esno_lin / (M - 1)));

else
    ber = 0.5 * ones(size(esno_lin));
end

ber = min(max(ber, 1e-12), 0.5);
end