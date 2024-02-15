function [] = two_layer_mlp(x_train, z_train, x_test, z_test)
    T = 50;
    mse_train = zeros(1, T);
    mse_test = zeros(1, T);
    N = 10;
    neuron_count = 1:T; % Neuron count 

for M = 1:T
    for i = 1:N
        net = feedforwardnet([5 M]);
        net = init(net);
        net.trainFcn = 'trainlm'; %  or use "traingd or trainscg " instead
        net.divideParam.trainRatio = 1;
        net.divideParam.testRatio = 0;
        net.divideParam.valRatio = 0;
        net.performFcn = 'mse';
        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'tansig'; % Second hidden layer transfer function
        net.layers{3}.transferFcn = 'purelin';
        net.trainParam.epochs = 30;
        net.trainParam.goal = 0.0001;
        net = train(net, x_train, z_train);
        
        y_train = sim(net, x_train);
        mse_train(M) = mse_train(M) + mse(z_train - y_train);
        
        y_test = sim(net, x_test);
        mse_test(M) = mse_test(M) + mse(z_test - y_test);
    end
    
    mse_test(M) = (1 / N) * mse_test(M);
    mse_train(M) = (1 / N) * mse_train(M);
end

    figure(2);
    plot(neuron_count, mse_train);
    hold on
    plot(neuron_count, mse_test, 'r');
    xlabel('Number of Neurons');
    ylabel('Mean Squared Error');
    legend('Training Error', 'Testing Error');
    title('Error vs. Number of Neurons');

    [~, optimal_neurons] = min(mse_test);
    fprintf('Optimal number of neurons: %d\n', optimal_neurons);


        net = feedforwardnet([5 optimal_neurons]);
       net = init(net);
        net.trainFcn = 'trainlm'; %  or use "traingd or trainscg " instead
        net.divideParam.trainRatio = 1;
        net.divideParam.testRatio = 0;
        net.divideParam.valRatio = 0;
        net.performFcn = 'mse';
        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'tansig'; % Second hidden layer transfer function
        net.layers{3}.transferFcn = 'purelin';
        net.trainParam.epochs = 30;
        net.trainParam.goal = 0.0001;
        net = train(net, x_train, z_train);
        y_pred = sim(net, x_test);
        
        figure(4);
    hold on
    plot(z_test);
    hold on
    plot(y_pred, 'r');
    xlabel('Sample');
    ylabel('Output');
    legend('Desired Output', 'Predicted Output');
    title('Desired Output vs. Predicted Output');

end
        
        
    
    
    
    
    
    
    
    

