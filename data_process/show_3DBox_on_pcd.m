% load lidar data
lidarData = pcread('ouster1_300.pcd');

% define 3D bouding box 
% 3d_box = [1.8184e+01; 4.5109e+01; -2.7786e+01; 5.1105e+00; -1.5019e+01;5.1105e+00];
% x_center = (1.8184e+01 + 5.1105e+00)/2;
% y_center = (4.5109e+01 + -1.5019e+01)/2;
% z_center = (-2.7786e+01 + 5.1105e+00)/2;
% length   = abs(1.8184e+01 - 5.1105e+00);
% width    = abs(4.5109e+01 - -1.5019e+01);
% height   = abs(-2.7786e+01 - 5.1105e+00);
% yaw      = 0;
% define 3D bounding box
x_min = 1.8184e+01;
x_max = 5.1105e+00;
y_min = 4.5109e+01;
y_max = -1.5019e+01;
z_min = -2.7786e+01;
z_max = 5.1105e+00;

x_center = (x_min + x_max) / 2;
y_center = (y_min + y_max) / 2;
z_center = (z_min + z_max) / 2;
length   = abs(x_max - x_min);
width    = abs(y_max - y_min);
height   = abs(z_max - z_min);
yaw      = 0; % Assuming no rotation for simplicity
bboxes = [x_center, y_center, z_center, length, width, height, yaw];

% 可视化点云
figure;
pcshow(lidarData);
hold on;

% % 添加边界框到可视化中
% for i = 1:size(bboxes, 1)
%     center = bboxes(i, 1:3);
%     dimensions = bboxes(i, 4:6);
%     orientation = axang2quat([0 0 1 bboxes(i, 7)]);
%     cuboid = alphaShape([center; center + dimensions]);
%     plot(cuboid, 'FaceColor', 'red', 'FaceAlpha', 0.5,'LineWidth',100);hold on;
% end
% add bounding box to visualization
for i = 1:size(bboxes, 1)
    center = bboxes(i, 1:3);
    dimensions = bboxes(i, 4:6);
    
    % Define the vertices of the cuboid
    vertices = [center(1) - dimensions(1)/2, center(2) - dimensions(2)/2, center(3) - dimensions(3)/2;
                center(1) + dimensions(1)/2, center(2) - dimensions(2)/2, center(3) - dimensions(3)/2;
                center(1) + dimensions(1)/2, center(2) + dimensions(2)/2, center(3) - dimensions(3)/2;
                center(1) - dimensions(1)/2, center(2) + dimensions(2)/2, center(3) - dimensions(3)/2;
                center(1) - dimensions(1)/2, center(2) - dimensions(2)/2, center(3) + dimensions(3)/2;
                center(1) + dimensions(1)/2, center(2) - dimensions(2)/2, center(3) + dimensions(3)/2;
                center(1) + dimensions(1)/2, center(2) + dimensions(2)/2, center(3) + dimensions(3)/2;
                center(1) - dimensions(1)/2, center(2) + dimensions(2)/2, center(3) + dimensions(3)/2];
    
    % Define the faces of the cuboid
    faces = [1 2 3 4;
             5 6 7 8;
             1 2 6 5;
             2 3 7 6;
             3 4 8 7;
             4 1 5 8];
    
    % Create a patch object for each cuboid
    patch('Vertices', vertices, 'Faces', faces, 'FaceColor', 'red', 'FaceAlpha', 0.5);
end

% hold off;


%%
% 设置点云数据文件夹和视频输出路径
dataFolder = 'path_to_your_lidar_data_folder';
videoOutputPath = 'output_video_path.avi';

% 创建 VideoWriter 对象
v = VideoWriter(videoOutputPath);
open(v);

% 定义边界框信息（示例）
% 您需要根据实际情况替换或计算边界框数据
bboxes = [x_center, y_center, z_center, length, width, height, yaw];

% 获取点云文件列表
filePattern = fullfile(dataFolder, '*.pcd');
lidarFiles = dir(filePattern);

% 循环遍历所有点云文件
for k = 1:length(lidarFiles)
    baseFileName = lidarFiles(k).name;
    fullFileName = fullfile(dataFolder, baseFileName);
    
    % 读取点云数据
    ptCloud = pcread(fullFileName);
    
    % 创建图形并显示点云
    f = figure('visible', 'off');
    pcshow(ptCloud);
    hold on;
    
    % 添加边界框到可视化中
    for i = 1:size(bboxes, 1)
        center = bboxes(i, 1:3);
        dimensions = bboxes(i, 4:6);
        orientation = axang2quat([0 0 1 bboxes(i, 7)]);
        cuboid = alphaShape([center; center + dimensions]);
        plot(cuboid, 'FaceColor', 'red', 'FaceAlpha', 0.5);
    end
    
    hold off;
    
    % 捕获当前图形并写入视频
    frame = getframe(gcf);
    writeVideo(v, frame);
    
    % 关闭图形窗口以节省资源
    close(f);
end

% 关闭视频文件
close(v);
