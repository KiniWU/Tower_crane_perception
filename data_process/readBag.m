%https://www.mathworks.com/matlabcentral/answers/1914840-how-to-convert-bag-file-into-pcd-in-matlab-lidar-data
bagMsgs = rosbagreader("/home/haochen/HKCRC/tower_crane_data/site_data/test3/_2024-05-02-11-09-59_19.bag");

%%
camera1_Msgs = select(bagMsgs,Topic='/camera1/image_com/compressed');
ouster_pointMsgs  = select(bagMsgs,Topic='/ouster/points');
ouster_imuMsgs   = select(bagMsgs,Topic='/ouster/imu');
% camera2_Msgs = select(bagMsgs,Topic='/camera2/image_com/compressed');
% ouster2_imuMsgs   = select(bagMsgs,Topic='/ouster2/imu');

camera1_Msgs = readMessages(camera1_Msgs,'DataFormat','struct');
ouster_pointMsgs = readMessages(ouster_pointMsgs,'DataFormat','struct');
ouster_imuMsgs   = readMessages(ouster_imuMsgs,'DataFormat','struct');

% camera2_Msgs = readMessages(camera2_Msgs,'DataFormat','struct');
% ouster2_imuMsgs   = readMessages(ouster2_imuMsgs,'DataFormat','struct');

%%
outputFolder = 'ouster';

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

for i=1:length(ouster_pointMsgs)
    ptCloud = rosReadXYZ(ouster_pointMsgs{i});
    ptCloud = pointCloud(ptCloud);
    filename = fullfile(outputFolder,sprintf('ouster_%d.pcd',i));
    pcwrite(ptCloud,filename);
end
%%
outputFolder = 'camera1';
mkdir(outputFolder);


for i=1:length(camera1_Msgs)
    image_out = rosReadImage(camera1_Msgs{i});
    filename = fullfile(outputFolder,sprintf('camera1_%d.png',i));
    imwrite(image_out,filename);
end

%%
outputFolder = 'camera2';
mkdir(outputFolder);


for i=1:length(camera2_Msgs)
    image_out = rosReadImage(camera2_Msgs{i});
    filename = fullfile(outputFolder,sprintf('camera2_%d.png',i));
    imwrite(image_out,filename);
end