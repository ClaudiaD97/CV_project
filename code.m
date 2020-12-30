% IMAGE CLASSIFIER BASED ON CNN
close all force 

% set the seed of the random number generator for reproducibility
rng(0)

% create training set and test set
TrainDatasetPath = fullfile('Images','train');
trainData = imageDatastore(TrainDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');

% classes and images per class
unique(trainData.Labels)
countEachLabel(trainData)

% print an image
iimage=1;
img = trainData.readimage(iimage); 
figure;
imshow(img,'initialmagnification',1000)

% input image should be 64x64, apply anisotropic rescaling
img_res = imresize(img,[64,64]);
size(img_res)
figure
imshow(img_res,'initialmagnification',1000)

% in this way automatic resizing when image is accessed (also values in 0-1 and not 0-255)
%trainData.ReadFcn = @(x)double(imresize(imread(x),[64 64]))/256;

% to reset the function execute next line
% trainData.ReadFcn = @(x)imread(x);

% now divide real training set and validation set
quota_training=0.85;
[trainingSet,validationSet] = splitEachLabel(trainData,quota_training,'randomize');

% augment just training set (optional 4)
trainingSet_aug = transform(trainingSet,@randomCropAndResize);
% merge with training set
ts = combine(trainingSet,trainingSet_aug);
%preview(ts)
dataOut = read(ts)
figure
imshow(imtile(dataOut));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTIONS

% data augmentation point 2
function dataOut = addNoise(data)

dataOut = data;
for idx = 1:size(data,1)
   dataOut(idx) = imnoise(data(idx),'salt & pepper');
end

end

% data augmentation for optional4
function new_data = randomCropAndResize(data)
    crop_size = 64;
    new_data = data;
    class(data)
    for i=1:size(data,1)
        % random dimension
        dimension = size(data{1})
        xsize = dimension(1)
        ysize = dimension(2)
        max_dim = min(xsize,ysize)
        min_dim = crop_size
        new_size = randi([min_dim,max_dim],1)

        % top-left corner
        % x-coordinate in range 1, size(1)-64+1
        xnew = randi([1,xsize-new_size+1]);
        ynew = randi([1,ysize-new_size+1]);
        % crop and resize
        new_data(i) = imcrop(data(i),[xnew,ynew,new_size,new_size]);
        new_data(i) = imresize(new_data(i),[64,64]); % this is isotropic because already a square
    end
    % also small rotation?
end

% check this for data augmentation
% https://it.mathworks.com/help/deeplearning/ug/image-to-image-regression-using-deep-learning.html