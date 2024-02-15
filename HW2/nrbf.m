% % x_train = zscore(x_train);
% % x_test = zscore(x_test);
% % z_train = zscore(output_train1);
% % z_test = zscore(output_test1);
% % 
%NRBF parameters
numInputNodes = size(x_train, 1);
numOutputNodes = size(z_train, 1);
spread = 1; 

maxEpochs = 100;  
learningRate = 0.01;  

numHiddenNodesList = 1:1:15;
trainMSE = zeros(size(numHiddenNodesList));
testMSE = zeros(size(numHiddenNodesList));
bestNumHiddenNodes = 0;
bestMSE = Inf;

for idx = 1:numel(numHiddenNodesList)
    numHiddenNodes = numHiddenNodesList(idx);
    % NRBF initialization
    weights = randn(numHiddenNodes, numOutputNodes);
    centers = randn(numInputNodes, numHiddenNodes);
    biases = randn(1, numOutputNodes);
    % Training loop
    for epoch = 1:maxEpochs
        % Forward pass
        hiddenActivations = exp(-pdist2(x_train', centers', 'squaredeuclidean') / (2 * spread^2));
        normalizedHiddenActivations = hiddenActivations ./ sum(hiddenActivations, 2);  % Normalization
        output = normalizedHiddenActivations * weights + biases;
        % Backpropagation
        outputError = z_train - output;
        deltaWeights = learningRate * normalizedHiddenActivations' * outputError;
        deltaBiases = learningRate * sum(outputError);
        weights = weights + deltaWeights;
        biases = biases + deltaBiases;
    end
    hiddenActivations_train = exp(-pdist2(x_train', centers', 'squaredeuclidean') / (2 * spread^2));
    normalizedHiddenActivations_train = hiddenActivations_train ./ sum(hiddenActivations_train, 2);  % Normalization
    output_train = (normalizedHiddenActivations_train * weights + biases)';
    trainMSE(idx) = mean((z_train - output_train).^2, 'all');
    hiddenActivations_test = exp(-pdist2(x_test', centers', 'squaredeuclidean') / (2 * spread^2));
    normalizedHiddenActivations_test = hiddenActivations_test ./ sum(hiddenActivations_test, 2);  % Normalization
    output_test = (normalizedHiddenActivations_test * weights + biases)';
    testMSE(idx) = mean((z_test - output_test).^2, 'all');
    if testMSE(idx) < bestMSE
        bestMSE = testMSE(idx);
        bestNumHiddenNodes = numHiddenNodes;
        bestOutputTrain = output_train;
    end
end

figure;
plot(numHiddenNodesList, trainMSE, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
plot(numHiddenNodesList, testMSE, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6);
hold off;
grid on;
xlabel('Number of Hidden Nodes');
ylabel('MSE');
title('Test-Train MSE Error');
legend('Train MSE', 'Test MSE');

bestIdx = find(testMSE == bestMSE, 1);
hold on;
plot(numHiddenNodesList(bestIdx), bestMSE, 'go', 'LineWidth', 1.5, 'MarkerSize', 8);
hold off;

figure;
plot(z_train, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
plot(bestOutputTrain, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6);
hold off;
grid on;
xlabel('Sample');
ylabel('Value');
title('z\_train vs. Estimated Output');
legend('z\_train', 'Estimated Output');




%%%%%%%%%second method
% 
% % NRBF parameters
% numInputNodes = size(x_train, 1);
% numOutputNodes = size(z_train, 1);
% spread = 1; 
% 
% numHiddenNodesList = 1:1:15;
% trainMSE = zeros(size(numHiddenNodesList));
% testMSE = zeros(size(numHiddenNodesList));
% bestNumHiddenNodes = 0;
% bestMSE = Inf;
% 
% for idx = 1:numel(numHiddenNodesList)
%     numHiddenNodes = numHiddenNodesList(idx);
%     % NRBF initialization
%     centers = randn(numInputNodes, numHiddenNodes);
%     biases = randn(numOutputNodes, 1);
%     % Calculate the hidden activations
%     hiddenActivations_train = exp(-pdist2(x_train', centers', 'squaredeuclidean') / (2 * spread^2));
%     normalizedHiddenActivations_train = hiddenActivations_train ./ sum(hiddenActivations_train, 2);  % Normalization
%     % Solve the normal equations to find the optimal weights
%     H = [normalizedHiddenActivations_train, ones(size(normalizedHiddenActivations_train, 1), 1)];
%     weightsBiases = (H' * H) \ (H' * z_train');
%     weights = weightsBiases(1:end-1, :);
%     biases = weightsBiases(end, :);    
%     % Compute the output for training data
%     output_train = (normalizedHiddenActivations_train * weights) + biases';
%     trainMSE(idx) = mean((z_train - output_train).^2, 'all');  
%     % Compute the output for testing data
%     hiddenActivations_test = exp(-pdist2(x_test', centers', 'squaredeuclidean') / (2 * spread^2));
%     normalizedHiddenActivations_test = hiddenActivations_test ./ sum(hiddenActivations_test, 2);  % Normalization
%     output_test = (normalizedHiddenActivations_test * weights) + biases';
%     testMSE(idx) = mean((z_test - output_test).^2, 'all');
%     if testMSE(idx) < bestMSE
%         bestMSE = testMSE(idx);
%         bestNumHiddenNodes = numHiddenNodes;
%         bestOutputTrain = output_train;
%     end
% end
% 
% figure;
% plot(numHiddenNodesList, trainMSE, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
% hold on;
% plot(numHiddenNodesList, testMSE, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6);
% hold off;
% grid on;
% xlabel('Number of Hidden Nodes');
% ylabel('MSE');
% title('Test-Train MSE Error');
% legend('Train MSE', 'Test MSE');
% 
% bestIdx = find(testMSE == bestMSE, 1);
% hold on;
% plot(numHiddenNodesList(bestIdx), bestMSE, 'go', 'LineWidth', 1.5, 'MarkerSize', 8);
% hold off;
% 
% figure;
% plot(z_train, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
% hold on;
% plot(bestOutputTrain, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6);
% hold off;
% grid on;
% xlabel('Sample');
% ylabel('Value');
% title('z\_train vs. Estimated Output');
% legend('z\_train', 'Estimated Output');
% 
% 
% 
% 
% 
% 
























