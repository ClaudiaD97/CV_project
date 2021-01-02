
%%%%%%% IMAGE CLASSIFIER BASED ON CNN
close all force 

% set the seed of the random number generator for reproducibility
%rng(0)

%% 1) BASELINE

% create training set and test set
TrainDatasetPath = fullfile('Images','train');
trainData = imageDatastore(TrainDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
TestDatasetPath = fullfile('Images','test');
testData = imageDatastore(TestDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');

% classes and images per class
unique(trainData.Labels)
countEachLabel(trainData)

% print an image
iimage=1;
img = trainData.readimage(iimage); 
%figure;
%imshow(img,'initialmagnification',1000)

% in this way automatic resizing when image is accessed (also values in 0-1 and not 0-255)
%trainData.ReadFcn = @(x)imresize(double(imread(x))/255,[64 64]);
trainData.ReadFcn = @(x)imresize(imread(x),[64 64]);

% to reset the function execute next line
% trainData.ReadFcn = @(x)imread(x);
% important to do the same for TestSet
testData.ReadFcn = @(x)imresize(imread(x),[64,64]);

% now divide real training set and validation set
quota_training=0.85;
[trainingSet,validationSet] = splitEachLabel(trainData,quota_training,'randomize');

% augment training set
dataAugmenter = imageDataAugmenter('RandXReflection',true);
augTrainingSet = augmentedImageDatastore([64,64,1],trainingSet,'DataAugmentation', dataAugmenter)

layers = [
    imageInputLayer([64 64 1],'Name','input')
    
    convolution2dLayer(3,8,'WeightsInitializer','narrow-normal','BiasInitializer', 'zeros','Name','conv_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'WeightsInitializer', 'narrow-normal', 'BiasInitializer', 'zeros','Name','conv_2')
    reluLayer('Name','relu_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
 
    convolution2dLayer(3,32,'WeightsInitializer', 'narrow-normal', 'BiasInitializer', 'zeros','Name','conv_3')
    reluLayer('Name','relu_3')
   
    fullyConnectedLayer(15,'WeightsInitializer', 'narrow-normal', 'BiasInitializer', 'zeros','Name','fc_1')
    
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];

options = trainingOptions('sgdm', ... % method is stochastic gradient descent with momentum
    'ValidationData',validationSet, ... % which are validation data
    'MiniBatchSize',32, ... %power of 2 to exploit gpu
    'ExecutionEnvironment','parallel',... % parallel execution
    'Plots','training-progress') %show plots during training


% train the network
%net = trainNetwork(trainingSet,layers,options);

% to evaluate accuracy on test set use this function defined below
%[predictions, confusionMatrix, testAccuracy] = evaluateOnTestSet(net,testData)


%% 3) TRANSFER LEARNING

pretrainedNet = alexnet;
analyzeNetwork(pretrainedNet)
% take all layers except last 3 (fully connected, softmax, output)
pretrainedLayers = pretrainedNet.Layers(1:end-3);

% input is 227x227x3 so I need to adjust images
pretrainedLayers(1)
trainData.ReadFcn = @(x)readTrain(x);
[trainingSet,validationSet] = splitEachLabel(trainData,quota_training,'randomize');

%try to read an image
img = trainingSet.readimage(1); 
%figure
%imshow(img,'InitialMagnification',1000)
% ok it works

% and compose the new net architecture
layers = [ 
    pretrainedLayers
    fullyConnectedLayer(15,'Name','last_fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];

%training  options
options = trainingOptions('sgdm', ... 
    'ValidationData',validationSet, ...
    'MiniBatchSize',32, ... 
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress');

%actual training
fineTunedNet = trainNetwork(trainingSet,layers,options)
[predictions, confusionMatrix, testAccuracy] = evaluateOnTestSet(net,testData);


% 4) AUGMENT DATASET 

% I don't use dataaugmenter because applies transformation just with
% probability p
% with flip i can perform reflection
img2 = flip(img,2);
figure
imshow(img2,'initialmagnification',1000)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTIONS

% test accuracy
function[YPredicted, C, accuracy] = evaluateOnTestSet(net,testSet)
    YPredicted = classify(net,testSet);
    YTest = testSet.Labels;

    accuracy = sum(YPredicted == YTest)/numel(YTest);
    C = confusionmat(YPredicted, YTest);
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

function[I] = readTrain(x)
    img = imresize(imread(x),[227,227]);
    I = cat(3,img,img,img);
end

% check this for data augmentation
% https://it.mathworks.com/help/deeplearning/ug/image-to-image-regression-using-deep-learning.html