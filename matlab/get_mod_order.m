function M = get_mod_order(mod_str)
% GET_MOD_ORDER  从调制名称字符串中提取整数阶数
% 例: '16APSK' -> 16, '64QAM' -> 64, 'QPSK' -> 4, '8PSK' -> 8

tokens = regexp(mod_str, '(\d+)', 'tokens');
if isempty(tokens)
    % QPSK 特殊处理
    if strcmpi(mod_str, 'QPSK')
        M = 4;
    else
        M = 2;
    end
else
    M = str2double(tokens{1}{1});
end
end