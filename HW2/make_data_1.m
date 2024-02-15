q=zeros(2,192);
d=5; %number of zero elements in the matrix q
while d>0
%     [x_train,x_test]=data(121,71,2,-10,10);
    n_train = 121; % Number of training samples
    n_test = 71; % Number of testing samples
    l_min = -10;  % Minimum value of range
    l_max = 10;  % Maximum value of range
    m = 2; % Number of dimensions
    x_a = (l_max - l_min) * rand(m, n_train + n_test) + l_min * ones(m, n_train + n_test);
    x_test = zeros(m, n_test);
    x_train = zeros(m, n_train);

    ss = zeros(1, n_test); 
    % Randomly select n_test samples for test data
    for i = 1:n_test
        s = randi(n_train + n_test, [1, 1]);
    while size(find(ss == s)) == [1 1]
        s = randi(n_train + n_test, [1, 1]);
    end
    ss(i) = s;
    end

    for i = 1:n_test
        x_test(:, i) = x_a(:, ss(i));
    end    

    ss = sort(ss);
    s = 0;
    x_train(:, 1:ss(1) - 1) = x_a(:, 1:ss(1) - 1);
    for i = 1:n_test - 1
        x_train(:, ss(i) + 1 - i:ss(i + 1) - 1 - i) = x_a(:, ss(i) + 1:ss(i + 1) - 1);
    end
    
    q=[x_train,x_test];
    d=size(find(q==0));
    d=d(1);
end
z_train=zeros(1,121);
z_test=zeros(1,71);

variance_low = 0.001;
variance_med = 0.01;
variance_high = 0.1;

n_low_train = sqrt(variance_low) * randn(1, 121);
n_med_train = sqrt(variance_med) * randn(1, 121);
n_high_train = sqrt(variance_high) * randn(1, 121);

n_low_test=sqrt(variance_low)*randn(1,71);
n_med_test=sqrt(variance_med)*randn(1,71);
n_high_test=sqrt(variance_high)*randn(1,71);


for j=1:121
        z_train(1,j)=sin(x_train(1,j))*sin(x_train(2,j))/(x_train(1,j)*x_train(2,j));
end

output_train1 = z_train;
z_train_n_low=z_train+n_low_train;
z_train_n_med=z_train+n_med_train;
z_train_n_high=z_train+n_high_train;

for j=1:71
        z_test(1,j)=sin(x_test(1,j))*sin(x_test(2,j))/(x_test(1,j)*x_test(2,j));
end

output_test1 = z_test;
z_test_n_low=z_test+n_low_test;
z_test_n_med=z_test+n_med_test;
z_test_n_high=z_test+n_high_test;





