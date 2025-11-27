% 读取矩阵A
load('MatrixA.mat');
[m,n] = size(A);
Q = zeros(m,n);
R = zeros(n,n);
for k = 1:n
    % 计算R的第k列前k-1个元素（Q的前k-1列与A第k列的内积）
    R(1:k-1,k) = Q(:,1:k-1)' * A(:,k);
    % 从A第k列中减去Q前k-1列的投影，得到正交分量v
    v = A(:,k) - Q(:,1:k-1) * R(1:k-1,k);
    % R的对角元为v的模长
    R(k,k) = norm(v);
    % Q的第k列为v单位化后的向量
    Q(:,k) = v / R(k,k);
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
