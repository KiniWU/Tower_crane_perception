outputFolder_ROI = 'ouster_501_600_ROI';
if ~exist(outputFolder_ROI,'dir')
    mkdir(outputFolder_ROI);
end

namelist = dir('ouster_501_600\*.pcd');
len = length(namelist);
for i = 1:len
    file_name_1 = namelist(i).name;
    file_name_2 = fullfile("ouster_501_600\",file_name_1);
    ptCloud = pcread(file_name_2);
    
    xMin = -26.95;     % Minimum value along X-axis.
    yMin = -89.66;  % Minimum value along Y-axis.
    zMin = -61.74;    % Minimum value along Z-axis.
    xMax = 115.60;   % Maximum value along X-axis.
    yMax = 61.75;   % Maximum value along Y-axis.
    zMax = 52.23;     % Maximum value along Z-axis.

    % Define point cloud parameters.
    roi = [xMin xMax yMin yMax zMin zMax];
    indices = findPointsInROI(ptCloud,roi);
    ptCloud_ROI = select(ptCloud,indices);
    filename_3 = fullfile(outputFolder_ROI,file_name_1);
    pcwrite(ptCloud_ROI,filename_3);
end