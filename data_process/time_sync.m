function [synced_camera_msgs, synced_lidar_msgs, synced_camera_timestamps, synced_lidar_timestamps]...
                               = syncCameraLidar(camera_msgs, lidar_msgs)
    % init
    synced_camera_msgs = {};
    synced_lidar_msgs = {};

    synced_lidar_timestamps =[];
    synced_camera_timestamps = [];

    % obyain lidar and camera timestamp
    for i =1:length(camera_msgs)
        camera_msgs{i}.Header.Stamp.Sec  = double(camera_msgs{i}.Header.Stamp.Sec);
        camera_msgs{i}.Header.Stamp.Nsec = double(camera_msgs{i}.Header.Stamp.Nsec);
        camera_timestamps(i) = camera_msgs{i}.Header.Stamp.Sec + ...
                                  camera_msgs{i}.Header.Stamp.Nsec * 10^-9;
    end
    
    for i =1:length(lidar_msgs)
        lidar_msgs{i}.Header.Stamp.Sec  = double(lidar_msgs{i}.Header.Stamp.Sec);
        lidar_msgs{i}.Header.Stamp.Nsec = double(lidar_msgs{i}.Header.Stamp.Nsec);
        lidar_timestamps(i)  = lidar_msgs{i}.Header.Stamp.Sec + ...
                                   lidar_msgs{i}.Header.Stamp.Nsec * 10^-9;
    end

    % trim early timestamps
    if camera_timestamps(1) > lidar_timestamps(1)
            camera_timestamps = camera_timestamps- camera_timestamps(1);
            lidar_timestamps  = lidar_timestamps - camera_timestamps(1);
            % lidar_timestamps  = lidar_timestamps(lidar_timestamps>=0.0);
    else
            camera_timestamps = camera_timestamps- lidar_timestamps(1);
            lidar_timestamps  = lidar_timestamps - lidar_timestamps(1);
            % camera_timestamps = camera_timestamps(camera_timestamps>=0.0);
    end

    % camera_timestamps = camera_timestamps- camera_timestamps(1);
    % lidar_timestamps  = lidar_timestamps - lidar_timestamps(1);

    % set time sync tolerance
    % tolerance = 0.02;

    % sync timestamp
    for i = 1:length(camera_timestamps)
        if camera_timestamps(i) >= 0
        % find cloest lidar timestamp to camera  timestamp
        [min_difference, idx] = min(abs(lidar_timestamps - camera_timestamps(i)));

        % if min_difference <= tolerance
        %     synced_camera_msgs{end+1} = camera_msgs{i};
        %     synced_lidar_msgs{end+1} = lidar_msgs{idx};
        % end

        synced_lidar_msgs{end+1}       = lidar_msgs{idx};
        synced_lidar_timestamps(end+1) = lidar_timestamps(idx);
        synced_camera_msgs{end+1}      = camera_msgs{i};
        synced_camera_timestamps(end+1)= camera_timestamps(i);
        end
    end

end


%% https://www.mathworks.com/matlabcentral/answers/1914840-how-to-convert-bag-file-into-pcd-in-matlab-lidar-data
bagMsgs = rosbagreader("_2024-05-02-11-09-59_19.bag");

camera1_Msgs = select(bagMsgs,Topic='/camera1/image_com/compressed');
ouster_pointMsgs  = select(bagMsgs,Topic='/ouster/points');
% ouster_imuMsgs   = select(bagMsgs,Topic='/ouster/imu');
% camera2_Msgs = select(bagMsgs,Topic='/camera2/image_com/compressed');
% ouster2_imuMsgs   = select(bagMsgs,Topic='/ouster2/imu');

camera1_Msgs = readMessages(camera1_Msgs,'DataFormat','struct');
ouster_pointMsgs = readMessages(ouster_pointMsgs,'DataFormat','struct');
% ouster_imuMsgs   = readMessages(ouster_imuMsgs,'DataFormat','struct');
% camera2_Msgs = readMessages(camera2_Msgs,'DataFormat','struct');
% ouster2_imuMsgs   = readMessages(ouster2_imuMsgs,'DataFormat','struct');

%%
[synced_camera_msgs, synced_lidar_msgs, synced_camera_timestamps, synced_lidar_timestamps]...
                         = syncCameraLidar(camera1_Msgs, ouster_pointMsgs);
%%
synced_camera_folder = 'sync_camera_lidar/camera1';
synced_lidar_folder  = 'sync_camera_lidar/ouster1';
if ~exist(synced_camera_folder,'dir')
    mkdir(synced_camera_folder);
end
if ~exist(synced_lidar_folder,'dir')
    mkdir(synced_lidar_folder);
end

for i=1:length(synced_lidar_msgs)
    ptCloud = rosReadXYZ(synced_lidar_msgs{i});
    ptCloud = pointCloud(ptCloud);
    filename = fullfile(synced_lidar_folder,sprintf('ouster1_%d.pcd',i));
    pcwrite(ptCloud,filename);
end


for i=1:length(synced_camera_msgs)
    image_out = rosReadImage(synced_camera_msgs{i});
    filename = fullfile(synced_camera_folder,sprintf('camera1_%d.png',i));
    imwrite(image_out,filename);
end

