%% 病态矩阵（希尔伯特矩阵）下 GS-QR vs Householder-QR 求逆性能测试
clear; clc; close all;

% 测试希尔伯特矩阵阶数（注意：n≥12时希尔伯特矩阵已高度病态）
n_list = [5, 10, 15, 20];  % 阶数不宜过高（n=20已足够体现病态性）
% 存储结果
gs_time = zeros(size(n_list));   % GS耗时
hh_time = zeros(size(n_list));   % Householder耗时
gs_err = zeros(size(n_list));    % GS求逆误差
hh_err = zeros(size(n_list));    % Householder求逆误差
cond_B = zeros(size(n_list));    % 矩阵条件数（衡量病态程度）

for i = 1:length(n_list)
    n = n_list(i);
    fprintf('测试 %d 阶希尔伯特矩阵...\n', n);
    
    % 生成病态矩阵：希尔伯特矩阵（列高度近似线性相关）
    B = hilb(n); 
    % 计算条件数（条件数越大，矩阵越病态）
    cond_B(i) = cond(B);
    
    % --------------------- GS-QR 求逆 ---------------------
    tic;
    [Q_gs, R_gs] = gram_schmidt_qr(B);
    R_gs_inv = upper_tri_inv(R_gs);
    B_inv_gs = R_gs_inv * Q_gs';
    gs_time(i) = toc;
    % 计算求逆误差：||B*B_inv - I||_2（2-范数）
    gs_err(i) = norm(B * B_inv_gs - eye(n));
    
    % --------------------- Householder-QR 求逆 ---------------------
    tic;
    [Q_hh, R_hh] = householder_qr(B);
    R_hh_inv = upper_tri_inv(R_hh);
    B_inv_hh = R_hh_inv * Q_hh';
    hh_time(i) = toc;
    hh_err(i) = norm(B * B_inv_hh - eye(n));
end

% --------------------- 结果输出 ---------------------
disp('=== 希尔伯特矩阵（病态）求逆测试结果 ===');
result_table = table(n_list', cond_B', gs_time', hh_time', gs_err', hh_err', ...
    'VariableNames', {'矩阵阶数', '条件数', 'GS耗时(s)', 'Householder耗时(s)', 'GS求逆误差', 'Householder求逆误差'});
disp(result_table);

% --------------------- 可视化 ---------------------
figure('Position', [100, 100, 1200, 500]);

% 子图1：条件数 vs 矩阵阶数（体现病态程度）
subplot(1,3,1);
semilogy(n_list, cond_B, 'o-', 'LineWidth', 1.5, 'Color', '#2E86AB');
xlabel('矩阵阶数'); ylabel('条件数（对数刻度）'); 
title('希尔伯特矩阵病态程度');
grid on;

% 子图2：耗时对比
subplot(1,3,2);
plot(n_list, gs_time, 'o-', 'LineWidth', 1.5, 'DisplayName', 'GS-QR');
hold on;
plot(n_list, hh_time, 's-', 'LineWidth', 1.5, 'DisplayName', 'Householder-QR');
xlabel('矩阵阶数'); ylabel('耗时 (s)'); title('耗时对比');
legend; grid on;

% 子图3：误差对比（对数刻度，突出GS误差爆炸）
subplot(1,3,3);
semilogy(n_list, gs_err, 'o-', 'LineWidth', 1.5, 'DisplayName', 'GS-QR');
hold on;
semilogy(n_list, hh_err, 's-', 'LineWidth', 1.5, 'DisplayName', 'Householder-QR');
xlabel('矩阵阶数'); ylabel('求逆误差（对数刻度）'); title('误差对比（病态矩阵）');
legend; grid on;

% --------------------- 复用之前的子函数 ---------------------
function [Q, R] = gram_schmidt_qr(A)
    [m,n] = size(A);
    Q = zeros(m,n); R = zeros(n,n);
    for k = 1:n
        R(1:k-1,k) = Q(:,1:k-1)' * A(:,k);
        v = A(:,k) - Q(:,1:k-1)*R(1:k-1,k);
        R(k,k) = norm(v);
        if R(k,k) < eps  % 避免除以0（病态矩阵易触发）
            warning('矩阵列近似线性相关，GS正交化误差将显著增大！');
            Q(:,k) = v / (R(k,k) + eps); % 避免除0
        else
            Q(:,k) = v/R(k,k);
        end
    end
end

function [Q, R] = householder_qr(A)
    [m,n] = size(A); Q = eye(m); R = A;
    for k = 1:n-1
        x = R(k:m,k); e1 = zeros(length(x),1); e1(1)=1;
        v = x + sign(x(1))*norm(x)*e1; 
        if norm(v) < eps
            v = e1; % 退化情况处理
        end
        v = v/norm(v);
        H = eye(m-k+1) - 2*v*v';
        R(k:m,k:n) = H*R(k:m,k:n);
        Q(:,k:m) = Q(:,k:m)*H;
    end
end

function R_inv = upper_tri_inv(R)
    [n,n] = size(R); R_inv = zeros(n,n);
    for k = 1:n
        x = zeros(n,1); 
        if abs(R(k,k)) < eps
            x(k) = 1/(R(k,k) + eps); % 病态矩阵对角线接近0时处理
        else
            x(k) = 1/R(k,k);
        end
        for i = k-1:-1:1
            x(i) = (-R(i,i+1:k)*x(i+1:k))/R(i,i);
        end
        R_inv(:,k) = x;
    end
end