% 读取图像
I = imread('your_image.jpg'); % 替换 'your_image.jpg' 为您的图像文件名
[M, N, ~] = size(I); % 获取图像的尺寸

% 确定每个块的目标大小
targetM = 8 * floor(M / 8);
targetN = 8 * floor(N / 8);

% 如果需要，调整图像大小
if M ~= targetM || N ~= targetN
    I = imresize(I, [targetM, targetN]);
end

% 更新图像尺寸
[M, N, ~] = size(I);

% 计算每个块的大小
m = M / 8;
n = N / 8;

% 初始化计数器
count = 1;

% 循环切割并保存图像块
for i = 1:8
    for j = 1:8
        % 计算当前块的索引
        row_start = (i-1) * m + 1;
        row_end = i * m;
        col_start = (j-1) * n + 1;
        col_end = j * n;
        
        % 切割图像块
        block = I(row_start:row_end, col_start:col_end, :);
        
        % 保存图像块
        filename = sprintf('block_%d.jpg', count);
        imwrite(block, filename);
        
        % 更新计数器
        count = count + 1;
    end
end


