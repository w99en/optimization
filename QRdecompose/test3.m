%% 矩阵B可逆性判断 + 多种方法求逆（直接求逆/GS-QR求逆/Householder-QR求逆）
clear; clc; close all;

% ===================== 1. 加载矩阵并基础校验 =====================
load('MatrixB.mat');  % 加载矩阵B（替换为你的矩阵路径/定义）
% 若没有MatrixB.mat，可手动构造测试矩阵（示例：3阶可逆方阵）
% B = randn(3); % 随机可逆方阵
% B = [1 2 3; 2 5 7; 3 6 8]; % 手动定义方阵

% 基础信息输出
disp('=== 矩阵B基础信息 ===');
if ismatrix(B)
    [m,n] = size(B);
    fprintf('矩阵B维度：%d行 × %d列\n', m, n);
else
    error('输入的不是有效矩阵！');
end

% ===================== 2. 可逆性判断 =====================
disp('\n=== 可逆性判断 ===');
if m ~= n  % 非方阵直接判定不可逆
    disp('结论：矩阵B不是方阵，无法求逆！');
else
    % 方法1：秩判断（更稳定，避免行列式数值误差）
    rank_B = rank(B);
    % 方法2：行列式辅助验证（仅参考，高维矩阵行列式误差大）
    det_B = det(B);
    
    fprintf('矩阵B的秩：%d，阶数：%d\n', rank_B, n);
    fprintf('矩阵B的行列式：%.4e\n', det_B);
    
    if rank_B == n  % 满秩=可逆
        disp('结论：矩阵B可逆，开始求逆...');
        
        % ===================== 3. 方法1：直接求逆（Matlab内置函数） =====================
        disp('\n=== 方法1：直接求逆（inv函数） ===');
        B_inv_direct = inv(B);
        % 验证直接求逆结果
        err_direct = norm(B * B_inv_direct - eye(n));
        fprintf('直接求逆验证误差（B*B_inv - I的范数）：%.4e\n', err_direct);
        
        % ===================== 4. 方法2：经典GS-QR分解求逆 =====================
        disp('\n=== 方法2：经典GS-QR分解求逆 ===');
        % 子函数：经典GS算法实现QR分解
        [Q_gs, R_gs] = gram_schmidt_qr(B);
        % 求上三角矩阵R的逆（回代法，比inv(R)更高效）
        R_gs_inv = upper_tri_inv(R_gs);
        % QR求逆核心公式：B_inv = R_inv * Q'
        B_inv_gs = R_gs_inv * Q_gs';
        % 验证GS-QR求逆结果
        err_gs = norm(B * B_inv_gs - eye(n));
        fprintf('GS-QR求逆验证误差：%.4e\n', err_gs);
        
        % ===================== 5. 方法3：Householder-QR分解求逆 =====================
        disp('\n=== 方法3：Householder-QR分解求逆 ===');
        % 子函数：Householder算法实现QR分解
        [Q_hh, R_hh] = householder_qr(B);
        % 求上三角矩阵R的逆
        R_hh_inv = upper_tri_inv(R_hh);
        % QR求逆核心公式
        B_inv_hh = R_hh_inv * Q_hh';
        % 验证Householder-QR求逆结果
        err_hh = norm(B * B_inv_hh - eye(n));
        fprintf('Householder-QR求逆验证误差：%.4e\n', err_hh);
        
        % ===================== 6. 结果汇总 =====================

        
    else
        disp('结论：矩阵B不可逆（秩 < 阶数），无法求逆！');
    end
end

% ===================== 子函数1：经典Gram-Schmidt QR分解 =====================
function [Q, R] = gram_schmidt_qr(A)
    [m,n] = size(A);
    Q = zeros(m,n);
    R = zeros(n,n);
    for k = 1:n
        % 正交化：消去前k-1列的投影
        R(1:k-1,k) = Q(:,1:k-1)' * A(:,k);
        v = A(:,k) - Q(:,1:k-1) * R(1:k-1,k);
        % 单位化
        R(k,k) = norm(v);
        if R(k,k) < eps  % 避免除以0（理论上可逆矩阵不会触发）
            error('矩阵列线性相关，无法完成GS-QR分解！');
        end
        Q(:,k) = v / R(k,k);
    end
end

% ===================== 子函数2：Householder QR分解 =====================
function [Q, R] = householder_qr(A)
    [m,n] = size(A);
    Q = eye(m);  % 初始化为单位矩阵
    R = A;       % 初始化为A的副本
    for k = 1:n-1
        % 提取当前列的子向量
        x = R(k:m, k);
        % 构造Householder向量
        e1 = zeros(length(x), 1);
        e1(1) = 1;
        v = x + sign(x(1)) * norm(x) * e1;
        v = v / norm(v);  % 单位化
        % 构造Householder矩阵并更新R、Q
        H = eye(m - k + 1) - 2 * v * v';
        R(k:m, k:n) = H * R(k:m, k:n);
        Q(:, k:m) = Q(:, k:m) * H;
    end
end

% ===================== 子函数3：上三角矩阵求逆（回代法） =====================
function R_inv = upper_tri_inv(R)
    [n,n] = size(R);
    R_inv = zeros(n,n);
    for k = 1:n
        % 求解R·x = e_k（e_k为第k个单位基向量）
        x = zeros(n,1);
        x(k) = 1 / R(k,k);  % 对角线元素回代
        % 上三角回代求解
        for i = k-1:-1:1
            x(i) = ( -R(i,i+1:k) * x(i+1:k) ) / R(i,i);
        end
        R_inv(:,k) = x;  % 填充逆矩阵的第k列
    end
end