%%
input_file_path  = '/home/haochen/HKCRC/tower_crane_data/site_data/test4/preprocess_data/sync_camera_lidar/livox'; % Replace with the path to your images folder
output_file_path = '/home/haochen/HKCRC/tower_crane_data/site_data/test4/preprocess_data/sync_camera_lidar/rotated_livox';
if ~exist(output_file_path,'dir')
    mkdir(output_file_path);
end

namelist = dir(fullfile(input_file_path,'*.pcd'));
len      = length(namelist);
R_ib     = eye(3);

for i = 1:len
    file_name = namelist(i).name;
    ptCloud   = pcread(fullfile(input_file_path, file_name));
    pt_b      = ptCloud.Location';
    pt_i      = R_ib* pt_b;
    rotated_ptCloud = pointCloud(pt_i');
    pcwrite(rotated_ptCloud,fullfile(output_file_path,file_name));
    print("process",i/len)
end