clc; 
clear all; 
close all; 

img = imread('b.png'); 
figure; imshow(img); title('Original Image'); 

% Convert the input image to double
img = double(img); 

% Normalize the input image to the range [0, 1]
img = img / 255; 

% Get the size of the input image
[row, col, dim] = size(img); 

% Reshape the input image into a 2D matrix
reshaped_img = reshape(img, [row*col, dim]); 

% Set the number of clusters
num_clusters = 5; 

% Initialize the centroids randomly
centroids = rand(num_clusters, dim); 

% Set the maximum number of iterations
max_iter = 100; 

% Iterate until convergence or until reaching the maximum number of iterations
for i = 1:max_iter
    
    % Initialize the distance matrix
    dist_mat = zeros(row*col, num_clusters); 
    
    % Compute the distances between each point and each centroid
    for j = 1:num_clusters
        centroid_mat = repmat(centroids(j,:), row*col, 1); 
        diff_mat = reshaped_img - centroid_mat; 
        dist_mat(:,j) = sum(diff_mat.^2, 2); 
    end
    
    % Assign each point to the closest centroid
    [~, label] = min(dist_mat, [], 2); 
    
    % Compute the new centroids
    for j = 1:num_clusters
        mask = (label == j); 
        if sum(mask) > 0
            centroids(j,:) = mean(reshaped_img(mask, :)); 
        end
    end
    
end

% Reshape the label matrix back into the original image size
label_mat = reshape(label, [row, col]); 

% Display the segmented image
segmented_img = zeros(row, col, dim); 
for i = 1:num_clusters
    mask = (label_mat == i); 
    segmented_img(:,:,1) = segmented_img(:,:,1) + mask * centroids(i,1); 
    segmented_img(:,:,2) = segmented_img(:,:,2) + mask * centroids(i,2); 
    segmented_img(:,:,3) = segmented_img(:,:,3) + mask * centroids(i,3); 
end
segmented_img = uint8(segmented_img * 255); 
figure; imshow(segmented_img); title('Segmented Image');
