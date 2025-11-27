%% GS-QR vs Householder-QR 求逆性能测试
clear; clc; close all;

% 测试矩阵阶数（可调整：50/100/200/500/1000）
n_list = [50, 100, 200, 500]; 
% 存储结果
gs_time = zeros(size(n_list));   % GS耗时
hh_time = zeros(size(n_list));   % Householder耗时
gs_err = zeros(size(n_list));    % GS求逆误差
hh_err = zeros(size(n_list));    % Householder求逆误差

for i = 1:length(n_list)
    n = n_list(i);
    fprintf('测试 %d 阶随机方阵...\n', n);
    
    % 生成可逆方阵（随机矩阵，条件数适中）
    B = randn(n); 
    % 可选：生成病态矩阵（放大稳定性差异）
    % B = hilb(n); % 希尔伯特矩阵，高条件数
    
    % --------------------- GS-QR 求逆 ---------------------
    tic;
    [Q_gs, R_gs] = gram_schmidt_qr(B);
    R_gs_inv = upper_tri_inv(R_gs);
    B_inv_gs = R_gs_inv * Q_gs';
    gs_time(i) = toc;
    % 计算求逆误差：||B*B_inv - I||_2
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
disp('=== 性能测试结果 ===');
result_table = table(n_list', gs_time', hh_time', gs_err', hh_err', ...
    'VariableNames', {'矩阵阶数', 'GS耗时(s)', 'Householder耗时(s)', 'GS求逆误差', 'Householder求逆误差'});
disp(result_table);

% --------------------- 可视化 ---------------------
figure('Position', [100, 100, 1000, 400]);
subplot(1,2,1);
plot(n_list, gs_time, 'o-', 'LineWidth', 1.5, 'DisplayName', 'GS-QR');
hold on;
plot(n_list, hh_time, 's-', 'LineWidth', 1.5, 'DisplayName', 'Householder-QR');
xlabel('矩阵阶数'); ylabel('耗时 (s)'); title('耗时对比');
legend; grid on;

subplot(1,2,2);
semilogy(n_list, gs_err, 'o-', 'LineWidth', 1.5, 'DisplayName', 'GS-QR');
hold on;
semilogy(n_list, hh_err, 's-', 'LineWidth', 1.5, 'DisplayName', 'Householder-QR');
xlabel('矩阵阶数'); ylabel('求逆误差（对数刻度）'); title('误差对比');
legend; grid on;

% --------------------- 复用之前的子函数 ---------------------
function [Q, R] = gram_schmidt_qr(A)
    [m,n] = size(A);
    Q = zeros(m,n); R = zeros(n,n);
    for k = 1:n
        R(1:k-1,k) = Q(:,1:k-1)' * A(:,k);
        v = A(:,k) - Q(:,1:k-1)*R(1:k-1,k);
        R(k,k) = norm(v);
        Q(:,k) = v/R(k,k);
    end
end

function [Q, R] = householder_qr(A)
    [m,n] = size(A); Q = eye(m); R = A;
    for k = 1:n-1
        x = R(k:m,k); e1 = zeros(length(x),1); e1(1)=1;
        v = x + sign(x(1))*norm(x)*e1; v = v/norm(v);
        H = eye(m-k+1) - 2*v*v';
        R(k:m,k:n) = H*R(k:m,k:n);
        Q(:,k:m) = Q(:,k:m)*H;
    end
end

function R_inv = upper_tri_inv(R)
    [n,n] = size(R); R_inv = zeros(n,n);
    for k = 1:n
        x = zeros(n,1); x(k) = 1/R(k,k);
        for i = k-1:-1:1
            x(i) = (-R(i,i+1:k)*x(i+1:k))/R(i,i);
        end
        R_inv(:,k) = x;
    end
end