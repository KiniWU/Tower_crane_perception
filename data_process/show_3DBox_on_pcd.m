% load lidar data
frame_number = 308;
lidarFolder = '/home/haochen/HKCRC/3D_object_detection/data/site_data/test3/sync_camera_lidar/ouster1';
lidarFilename = fullfile(lidarFolder,sprintf('ouster1_%d.pcd',frame_number));
lidarData     = pcread(lidarFilename);
% lidarData        = pcread('ouster1_300.pcd');
threed_boxes_all = load('/home/haochen/HKCRC/3D_object_detection/data/site_data/test3/sync_camera_lidar/threed_boxes.txt');
threed_boxes     = threed_boxes_all(frame_number,:);

% define 3D bounding box
% x_min = 1.8184e+01;
% x_max = 2.0467e+01;
% y_min = -2.7404e+01;
% y_max = -2.2677e+01;
% z_min = 1.0042e+01;
% z_max = 1.2742e+01;

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
%     plot(cuboid, 'FaceColor', 'red', 'FaceAlpha', 0.5,'LineWidth',1);hold on;
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


