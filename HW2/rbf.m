%%% Atterntion: see the dimenstion of data - make sure thet are correct

% x_train = zscore(x_train);
% x_test = zscore(x_test);
% z_train = zscore(z_train_n_low);
% z_test = zscore(z_test_n_low);


x_train = (x_train);
x_test = (x_test);
z_train = (z_train_n_med);
z_test = (z_test_n_med);

% Grid search for optimal number of neurons
numNeurons = 1:50;  %number of neurons in the middle layer
mse_train = zeros(size(numNeurons));
mse_test = zeros(size(numNeurons));

for i = 1:length(numNeurons)
    numClusters = numNeurons(i);
    % Step 1: K-means clustering
    [idx, centers] = kmeans(x_train', numClusters);
    % Step 2: RBF network training using newrb
    spread = 1;  % Spread parameter for the Gaussians
    net = newrb(x_train, z_train, 0, spread, numClusters);
    % Step 3: RBF network testing
    y_train = net(x_train); 
    y_test = net(x_test);    
    % Step 4: Performance evaluation
    mse_train(i,:) = mean((z_train - y_train).^2, 'all');  % Mean Squared Error for training data
    mse_test(i,:) = mean((z_test - y_test).^2, 'all');     % Mean Squared Error for testing data
end

% Find optimal number of neurons
[~, idxOptimal] = min(mse_test);
optimalNeurons = numNeurons(idxOptimal);
optimalMSE = mse_test(idxOptimal);

figure;
plot(numNeurons, mse_train, 'bo-', 'LineWidth', 2);
hold on;
plot(numNeurons, mse_test, 'ro-', 'LineWidth', 2);
plot(optimalNeurons, optimalMSE, 'gx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Number of Neurons');
ylabel('Mean Squared Error');
legend('Training MSE', 'Testing MSE', 'Optimal Neurons', 'Location', 'best');
title('Grid Search: Optimal Number of Neurons');

fprintf('Optimal number of neurons: %d\n', optimalNeurons);
fprintf('Optimal testing MSE: %f\n', optimalMSE);



 figure(4);hold on
    plot(z_test);
    hold on
    plot(y_test, 'r');
    xlabel('Sample');
    ylabel('Output');
    legend('Desired Output', 'Predicted Output');
    title('Desired Output vs. Predicted Output');
  

