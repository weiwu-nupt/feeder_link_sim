function ber = apsk_ber_approx(esno_lin, M)
% APSK_BER_APPROX  APSK 未编码 BER 近似（AWGN，Grey 编码）
%
% esno_lin : Es/No 线性值（数组）
% M        : 星座阶数（8/16/32/64/128/256）
% 返回 ber : 与 esno_lin 等长的 BER 数组

m = log2(M);

if M <= 8
    % 8APSK ≈ 等效 8PSK，外圆主导，修正因子 0.85
    ber = (1/m) * erfc(sqrt(esno_lin * 0.85 * (sin(pi/M))^2 * m));

elseif M == 16
    % 16APSK(4+12)，典型码率 3/4 的环比 r2/r1 = 2.57
    r1 = 1.0;
    r2 = 2.57;
    E  = r1^2 + r2^2;
    ser = (4/M)  * qfunc(sqrt(esno_lin) * r1 * sqrt(2/E) * sin(pi/4)) + ...
        (12/M) * qfunc(sqrt(esno_lin) * r2 * sqrt(2/E) * sin(pi/12));
    ber = max(ser / m, 1e-12);

elseif M == 32
    % 32APSK(4+12+16)，环比 r2/r1=2.53, r3/r1=4.3
    r1 = 1.0; r2 = 2.53; r3 = 4.3;
    E  = (4*r1^2 + 12*r2^2 + 16*r3^2) / 32;
    ser = (4/M)  * qfunc(sqrt(esno_lin/E) * r1 * sin(pi/4))  + ...
        (12/M) * qfunc(sqrt(esno_lin/E) * r2 * sin(pi/12)) + ...
        (16/M) * qfunc(sqrt(esno_lin/E) * r3 * sin(pi/16));
    ber = max(ser / m, 1e-12);

else
    % 64/128/256 APSK → 等效矩形 QAM 近似
    ber = (4/m) * (1 - 1/sqrt(M)) * qfunc(sqrt(3 * m * esno_lin / (M - 1)));
end

ber = min(max(ber, 1e-12), 0.5);
end