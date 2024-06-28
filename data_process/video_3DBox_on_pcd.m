

%%
% 设置点云数据文件夹和视频输出路径
% dataFolder = '/home/haochen/HKCRC/3D_object_detection/data/site_data/test3/sync_camera_lidar/ouster1';
% threed_boxes_all = load('/home/haochen/HKCRC/3D_object_detection/data/site_data/test3/sync_camera_lidar/threed_boxes.txt');
% videoOutputPath = 'ouster1_with_3DBox.avi';

dataFolder = '/home/haochen/HKCRC/3D_object_detection/data/site_data/test4/sync_camera_lidar/livox';
threed_boxes_all = load('/home/haochen/HKCRC/3D_object_detection/data/site_data/test4/sync_camera_lidar/threed_boxes.txt');
videoOutputPath = 'livox_with_3DBox.avi';

% 创建 VideoWriter 对象
v = VideoWriter(videoOutputPath);
v.FrameRate = 1;
open(v);

% 获取点云文件列表
filePattern = fullfile(dataFolder, '*.pcd');
lidarFiles = dir(filePattern);

% 循环遍历所有点云文件
% for k = 1:length(lidarFiles)
for k = 1:118
    baseFileName = lidarFiles(k).name;
    fullFileName = fullfile(dataFolder, baseFileName);
    
    % 读取点云数据
    ptCloud = pcread(fullFileName);
    
    % select ROI
    xMin = -26.95;     % Minimum value along X-axis.
    yMin = -89.66;  % Minimum value along Y-axis.
    zMin = -61.74;    % Minimum value along Z-axis.
    xMax = 115.60;   % Maximum value along X-axis.
    yMax = 61.75;   % Maximum value along Y-axis.
    zMax = 52.23;     % Maximum value along Z-axis.

    % Define point cloud parameters.
    roi = [xMin xMax yMin yMax zMin zMax];
    indices = findPointsInROI(ptCloud,roi);
    ptCloud = select(ptCloud,indices);
    
    % 创建图形并显示点云
    f = figure('visible', 'off');
    pcshow(ptCloud);
    hold on;

    % 定义边界框信息（示例）
    % 您需要根据实际情况替换或计算边界框数据
    threed_boxes     = threed_boxes_all(k,:);
    x_min = threed_boxes(1);
    x_max = threed_boxes(2);
    y_min = threed_boxes(3);
    y_max = threed_boxes(4);
    z_min = threed_boxes(5);
    z_max = threed_boxes(6);
    
    x_center = (x_min + x_max) / 2;
    y_center = (y_min + y_max) / 2;
    z_center = (z_min + z_max) / 2;
    length   = abs(x_max - x_min);
    width    = abs(y_max - y_min);
    height   = abs(z_max - z_min);
    yaw      = 0; % Assuming no rotation for simplicity
    bboxes = [x_center, y_center, z_center, length, width, height, yaw];
    
    % 添加边界框到可视化中
    % for i = 1:size(bboxes, 1)
    %     center = bboxes(i, 1:3);
    %     dimensions = bboxes(i, 4:6);
    %     orientation = axang2quat([0 0 1 bboxes(i, 7)]);
    %     cuboid = alphaShape([center; center + dimensions]);
    %     plot(cuboid, 'FaceColor', 'red', 'FaceAlpha', 0.5);
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
    
    hold off;
    
    % 捕获当前图形并写入视频
    frame = getframe(gcf);
    writeVideo(v, frame);
    
    % 关闭图形窗口以节省资源
    close(f);
end

% 关闭视频文件
close(v);