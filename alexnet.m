clear;
clc;
close all; 

imds = imageDatastore('ReDataset', 'IncludeSubfolders',true, 'LabelSource','foldernames');

[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, 0.6, 0.2, 0.2, 'randomized');

net = alexnet;
analyzeNetwork(net)

inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, 'DataAugmentation',imageAugmenter);

% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% [XTrain,YTrain] = digitTrain4DArrayData;
imageSize = [227 227 1];

% augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

augimdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'colorPreprocessing','gray2rgb', 'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(imageSize,imdsValidation,'colorPreprocessing','gray2rgb');
augimdsTest = augmentedImageDatastore(imageSize,imdsTest,'colorPreprocessing','gray2rgb');




options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',1, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

[YPredForValidation,scoresForValidation] = classify(netTransfer,augimdsValidation);
[YPredForTest,scoresForTest] = classify(netTransfer,augimdsTest);

idx = randperm(numel(imdsValidation.Files),4);
figure(1)
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPredForValidation(idx(i));
    title(string(label));
end

YValidation = imdsValidation.Labels;
accuracyForValidation = mean(YPredForValidation == YValidation);

YTest = imdsTest.Labels;
accuracyForTest = mean(YPredForTest == YTest);
plotconfusion(YTest, YPredForTest);
