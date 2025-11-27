load('MatrixA.mat');
[m,n] = size(A);
Q = eye(m); % 初始化为单位矩阵
R = A;
for k = 1:n-1
    % 提取当前列的子向量
    x = R(k:m, k);
    % 构造Householder向量
    e1 = zeros(length(x), 1);
    e1(1) = 1;
    v = x + sign(x(1)) * norm(x) * e1;
    v = v / norm(v); % 单位化
    % 构造Householder变换矩阵并更新R和Q
    H = eye(m - k + 1) - 2 * v * v';
    R(k:m, k:n) = H * R(k:m, k:n);
    Q(:, k:m) = Q(:, k:m) * H;
end


% ===================== 3. 稳定性验证（正交性退化程度） =====================
% 计算Q'Q与单位矩阵的误差范数（量化正交性退化）
ortho_error = norm(Q'*Q - eye(n)); % 误差范数越大，稳定性越差
fprintf('=== 稳定性验证 ===\n');
fprintf('Q''Q 与单位矩阵的误差范数：%.4e\n', ortho_error);
% 判定稳定性（机器精度参考：eps≈2.22e-16）
if ortho_error > 1e-10
    fprintf('结论：经典GS算法分解结果不稳定（正交性显著退化）\n');
else
    fprintf('结论：经典GS算法分解结果稳定（正交性退化在可接受范围）\n');
end

% ===================== 4. 正交性验证（严格正交性判断） =====================
fprintf('\n=== 正交性验证 ===\n');
fprintf('机器精度（eps）：%.4e\n', eps);
fprintf('Q''Q 与单位矩阵的误差范数：%.4e\n', ortho_error);
if ortho_error < 1e-15
    fprintf('结论：Q矩阵满足正交性（误差在机器精度范围内）\n');
else
    fprintf('结论：Q矩阵不满足正交性（误差超出机器精度范围）\n');
end

% ===================== 5. 辅助：输出Q'Q矩阵（直观查看偏离单位矩阵的程度） =====================
fprintf('\n=== Q''Q 矩阵（直观查看正交性） ===\n');
QtQ = Q'*Q;
disp(QtQ);
