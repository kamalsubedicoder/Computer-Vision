%the construction error is defined as a mean square error
%between the original image and the reconstructed image.
%The lower the mse is, the less we loss information in the
%compression/decompression process. 

%The compression rate is roughly CxC / (K) since CxC pixel
%is represented by using only K values and  (the space required
%to store the indices are ignored when calculating this 
%compression rate. 

image = imread("penguins.jpeg");

image_resized = imresize(image, [128, 128]);
image_resized_grayscale = rgb2gray(image_resized);

% Define values of k and c
k_values = [4, 8, 16];
c_values = [4, 8];

mse_values = zeros(length(k_values), length(c_values));

for k_idx = 1:length(k_values)
    k = k_values(k_idx);
    
    for c_idx = 1:length(c_values)
        c = c_values(c_idx);
        
        [height, width, ~] = size(image_resized);
        num_windows_h = floor(height / c);
        num_windows_w = floor(width / c);

        compressed_idx = cell(num_windows_h, num_windows_w);
        compressed_centers = cell(num_windows_h, num_windows_w);

        for i = 1:num_windows_h
            for j = 1:num_windows_w
                % C x C Window
                window = image_resized_grayscale((i-1)*c+1:i*c, (j-1)*c+1:j*c, :);

                [cluster_idx, cluster_centers] = kmeans(reshape(window, [], 1), k, 'MaxIter', 100);
                cluster_centers = uint8(cluster_centers);

                compressed_idx{i, j} = cluster_idx;
                compressed_centers{i, j} = cluster_centers;
            end
        end

        % Save the compressed data 
        compressed_data = {compressed_idx, compressed_centers};
        save(['compressed_data_k', num2str(k), '_c', num2str(c), '.mat'], 'compressed_data');

        %% Decompress the image using the compressed data
        reconstructed_image = zeros(size(image_resized_grayscale));

        for i = 1:num_windows_h
            for j = 1:num_windows_w
                
                load(['compressed_data_k', num2str(k), '_c', num2str(c), '.mat']);
                [compressed_idx, compressed_centers] = compressed_data{:};

                cluster_idx = compressed_idx{i, j};
                cluster_centers = compressed_centers{i, j};

                reconstructed_window = reshape(cluster_centers(cluster_idx, :), c, c);
                reconstructed_image((i-1)*c+1:i*c, (j-1)*c+1:j*c) = reconstructed_window;
            end
        end

        % Compute the MSE and store it in the mse_values matrix
        mse = immse(image_resized_grayscale, uint8(reconstructed_image));
        mse_values(k_idx, c_idx) = mse;

        
        figure;
        imshow(uint8(reconstructed_image));
        title(['Reconstructed Image (k=', num2str(k), ', c=', num2str(c), ')', ' MSE:', num2str(mse)]);
    end
end