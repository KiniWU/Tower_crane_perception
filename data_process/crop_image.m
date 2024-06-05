% Set the path to your images folder
input_file_path = '/home/haochen/HKCRC/3D_object_detection/data/site_data/test4/sync_camera_lidar/hikrobot_1_20'; % Replace with the path to your images folder
output_file_path = '/home/haochen/HKCRC/3D_object_detection/data/site_data/test4/sync_camera_lidar/croped_hikrobot_1_20';

% Get all images with .jpg format in the folder
img_path_list = dir(fullfile(input_file_path, '*.png'));
img_num = length(img_path_list); % Get the total number of images

% Check if there are any images
if img_num > 0
    for idx = 1:img_num
        % Read the image
        image_name = img_path_list(idx).name; % Image name
        I = imread(fullfile(input_file_path, image_name)); % Read the image
        
        % Adjust the image size to fit 8x8 cutting
        [M, N, ~] = size(I);
        targetM = 8 * floor(M / 8);
        targetN = 8 * floor(N / 8);
        if M ~= targetM || N ~= targetN
            I = imresize(I, [targetM, targetN]);
        end
        
        % Update image dimensions
        [M, N, ~] = size(I);
        
        % Calculate the size of each block
        m = M / 8;
        n = N / 8;
        
        % Initialize the counter
        count = 1;
        
        % Loop to cut and save image blocks
        for i = 1:8
            for j = 1:8
                % Calculate the index of the current block
                row_start = (i-1) * m + 1;
                row_end = i * m;
                col_start = (j-1) * n + 1;
                col_end = j * n;
                
                % Cut the image block
                block = I(row_start:row_end, col_start:col_end, :);
                
                % Save the image block
                filename = fullfile(output_file_path,sprintf('hikrobot%d_%d.jpg', idx, count));
                imwrite(block, filename);
                
                % Update the counter
                count = count + 1;
            end
        end
    end
end

