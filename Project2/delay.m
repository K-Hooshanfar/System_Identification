
%%%%%%%%%%%%%code for when we have delay - change the code according to the
%%%%%%%%%%%%% mlx files provided for each model

T = 1:400;
N = 10;
tmin = 25;
t0 = 1;
u = binrand(T, N, tmin, t0, 'normal');

% Define the transfer functions
numerator = [1, 0];
denominator = [10, -10];
ts = 0.01;
C = tf(numerator, denominator, ts);

num = [0.1812];
den = [1, -0.8187];
G = tf(num, den, ts);

delay=tf(1,[1 0 0 0 0 0 0 0 0 0 0], ts);

variance1 = 0.1; % also for 0.01 and 0.001
variance2 = 0.01;
variance3 = 0.001;
noise1 = sqrt(variance1) * randn(size(u));
noise2 = sqrt(variance2) * randn(size(u));
noise3 = sqrt(variance3) * randn(size(u));

Ts = 0.01;                % Sample time of the models
N = length(u); 
timeVector = 0:Ts:(N-1)*Ts;

output1 = lsim((G*C*delay)/(1+(G*C)), u.', timeVector.') + lsim(delay/(1+C*G), noise1.', timeVector.');
output2 = lsim((G*C*delay)/(1+(G*C)), u.', timeVector.') + lsim(delay/(1+C*G), noise2.', timeVector.');
output3 = lsim((G*C*delay)/(1+(G*C)), u.', timeVector.') + lsim(delay/(1+C*G), noise3.', timeVector.');

figure;
plot(output1)
hold on
plot(output2)
plot(output3)
plot(u)
legend('noise=0.1','noise=0.01', 'noise=0.001');
hold off

pairs1 = table(u.', output1, 'VariableNames', {'Input', 'Output'});
pairs2 = table(u.', output2, 'VariableNames', {'Input', 'Output'});
pairs3 = table(u.', output3, 'VariableNames', {'Input', 'Output'});

% Define a range of values for na, nb, and nk
% Parameters
na_values = [1, 2, 3 ,4 ,5];  % List of 'na' values to test
nc_values = [1, 2, 3 ,4,5];  % List of 'na' values to test
nb_values = [1, 2, 3 ,4,5];  % List of 'nb' values to test
nf_values = [1, 2, 3 ,4,5];  % List of 'na' values to test
nd_values = [1, 2, 3 ,4,5];  % List of 'na' values to test
nk_values = [15];  % List of 'nk' values to test
best_na = 0;            % Variable to store the best 'na' value
best_nb = 0;            % Variable to store the best 'nb' value
best_nk = 0;            % Variable to store the best 'nk' value
best_nc = 0;            % Variable to store the best 'nk' value
best_nf = 0;            % Variable to store the best 'nk' value
best_nd = 0;            % Variable to store the best 'nk' value
best_val_error = Inf;   % Variable to store the best validation error
best_model = [];        % Variable to store the best model



% Armax with 0.1 noise
best_y_pred = [];  
% Define a range of values for na, nb, and nk
best_na = 0;                % Variable to store the best 'na' value
best_nb = 0;                % Variable to store the best 'nb' value
best_nd = 0;                % Variable to store the best 'nd' value
best_nk = 0;                % Variable to store the best 'nk' value
best_val_error = Inf;       % Variable to store the best validation error
best_model = [];            % Variable to store the best model
best_y_pred = [];  
% Loop over each combination of 'na', 'nb', 'nd', and 'nk' values
for na = na_values
    for nb = nb_values
        for nd = nd_values
            for nk = nk_values
                % PEM Model Identification
                model = pem(iddata(output1, u.', 0.01), 'na', na, 'nb', nb, 'nd', nd, 'nk', nk);
                y_val_pred = sim(model, iddata(output1, u.', 0.01));
                val_error = mean((y_val_pred.OutputData - output1).^2);
                if val_error < best_val_error
                    best_val_error = val_error;
                    best_na = na;
                    best_nb = nb;
                    best_nd = nd;
                    best_nk = nk;
                    best_model = model;  % Store the best model
                    best_y_pred = y_val_pred.OutputData;
                end
            end
        end
    end
end

% Display the best parameter values and validation error
disp(['Best na value: ', num2str(best_na)]);
disp(['Best nb value: ', num2str(best_nb)]);
disp(['Best nd value: ', num2str(best_nd)]);
disp(['Best nk value: ', num2str(best_nk)]);
disp(['Best validation error (MSE): ', num2str(best_val_error)]);

% Display the best model
disp('Best model:');
disp(best_model);

% Plot the actual output and predicted output
figure;
t = 1:length(output1);
plot(t, output1, 'b--', 'LineWidth', 1.5);
hold on;
plot(t, best_y_pred, 'r', 'LineWidth', 1.5);
hold off;
xlabel('Time');
ylabel('Output');
legend('Actual Output', 'Predicted Output');
title('Comparison of Actual and Predicted Output');

best_model

% Extract zeros and poles from the estimated ARX model
zeros_est = zero(best_model);
poles_est = pole(best_model);

% Display the zero locations and poles
disp('Estimated Model:');
disp('Zeros:');
disp(zeros_est);
disp('Poles:');
disp(poles_est);

% Plot the zero-pole map
figure;
zplane(zeros_est, poles_est);
title('Zero-Pole Map');

e = u.' - best_y_pred;
corr_eu = corrcoef(e, u.');
corr_eu = corr_eu(1, 2); % Extract the correlation coefficient from the matrix

% Calculate the correlation between e and y
corr_ey = corrcoef(e, best_y_pred);
corr_ey = corr_ey(1, 2); % Extract the correlation coefficient from the matrix

% Display the correlation coefficients
disp('Correlation between e and u:');
disp(corr_eu);
disp('Correlation between e and y:');
disp(corr_ey);

corr_eu = xcorr(e, u.');

% Calculate the correlation between e and y
corr_ey = xcorr(e, best_y_pred);

% Plot the correlation results
lags = -(length(e)-1):(length(e)-1);
figure;
subplot(2, 1, 1);
stem(lags, corr_eu);
title('Correlation between e and u');
xlabel('Lag');
ylabel('Correlation');
subplot(2, 1, 2);
stem(lags, corr_ey);
title('Correlation between e and y');
xlabel('Lag');
ylabel('Correlation');


best_na = 0;                % Variable to store the best 'na' value
best_nb = 0;                % Variable to store the best 'nb' value
best_nd = 0;                % Variable to store the best 'nd' value
best_nk = 0;                % Variable to store the best 'nk' value
best_val_error = Inf;       % Variable to store the best validation error
best_model = [];            % Variable to store the best model
best_y_pred = [];  
% Loop over each combination of 'na', 'nb', 'nd', and 'nk' values
for na = na_values
    for nb = nb_values
        for nd = nd_values
            for nk = nk_values
                % PEM Model Identification
                model = pem(iddata(output2, u.', 0.01), 'na', na, 'nb', nb, 'nd', nd, 'nk', nk);
                y_val_pred = sim(model, iddata(output2, u.', 0.01));
                val_error = mean((y_val_pred.OutputData - output2).^2);
                if val_error < best_val_error
                    best_val_error = val_error;
                    best_na = na;
                    best_nb = nb;
                    best_nd = nd;
                    best_nk = nk;
                    best_model = model;  % Store the best model
                    best_y_pred = y_val_pred.OutputData;
                end
            end
        end
    end
end

% Display the best parameter values and validation error
disp(['Best na value: ', num2str(best_na)]);
disp(['Best nb value: ', num2str(best_nb)]);
disp(['Best nd value: ', num2str(best_nd)]);
disp(['Best nk value: ', num2str(best_nk)]);
disp(['Best validation error (MSE): ', num2str(best_val_error)]);

% Display the best model
disp('Best model:');
disp(best_model);

% Plot the actual output and predicted output
figure;
t = 1:length(output2);
plot(t, output2, 'b--', 'LineWidth', 1.5);
hold on;
plot(t, best_y_pred, 'r', 'LineWidth', 1.5);
hold off;
xlabel('Time');
ylabel('Output');
legend('Actual Output', 'Predicted Output');
title('Comparison of Actual and Predicted Output');

best_model

% Extract zeros and poles from the estimated ARX model
zeros_est = zero(best_model);
poles_est = pole(best_model);

% Display the zero locations and poles
disp('Estimated Model:');
disp('Zeros:');
disp(zeros_est);
disp('Poles:');
disp(poles_est);

% Plot the zero-pole map
figure;
zplane(zeros_est, poles_est);
title('Zero-Pole Map');

e = u.' - best_y_pred;
corr_eu = corrcoef(e, u.');
corr_eu = corr_eu(1, 2); % Extract the correlation coefficient from the matrix

% Calculate the correlation between e and y
corr_ey = corrcoef(e, best_y_pred);
corr_ey = corr_ey(1, 2); % Extract the correlation coefficient from the matrix

% Display the correlation coefficients
disp('Correlation between e and u:');
disp(corr_eu);
disp('Correlation between e and y:');
disp(corr_ey);

corr_eu = xcorr(e, u.');

% Calculate the correlation between e and y
corr_ey = xcorr(e, best_y_pred);

% Plot the correlation results
lags = -(length(e)-1):(length(e)-1);
figure;
subplot(2, 1, 1);
stem(lags, corr_eu);
title('Correlation between e and u');
xlabel('Lag');
ylabel('Correlation');
subplot(2, 1, 2);
stem(lags, corr_ey);
title('Correlation between e and y');
xlabel('Lag');
ylabel('Correlation');

% Define a range of values for na, nb, and nk

best_na = 0;                % Variable to store the best 'na' value
best_nb = 0;                % Variable to store the best 'nb' value
best_nd = 0;                % Variable to store the best 'nd' value
best_nk = 0;                % Variable to store the best 'nk' value
best_val_error = Inf;       % Variable to store the best validation error
best_model = [];            % Variable to store the best model
best_y_pred = [];  
% Loop over each combination of 'na', 'nb', 'nd', and 'nk' values
for na = na_values
    for nb = nb_values
        for nd = nd_values
            for nk = nk_values
                % PEM Model Identification
                model = pem(iddata(output3, u.', 0.01), 'na', na, 'nb', nb, 'nd', nd, 'nk', nk);
                y_val_pred = sim(model, iddata(output3, u.', 0.01));
                val_error = mean((y_val_pred.OutputData - output3).^2);
                if val_error < best_val_error
                    best_val_error = val_error;
                    best_na = na;
                    best_nb = nb;
                    best_nd = nd;
                    best_nk = nk;
                    best_model = model;  % Store the best model
                    best_y_pred = y_val_pred.OutputData;
                end
            end
        end
    end
end

% Display the best parameter values and validation error
disp(['Best na value: ', num2str(best_na)]);
disp(['Best nb value: ', num2str(best_nb)]);
disp(['Best nd value: ', num2str(best_nd)]);
disp(['Best nk value: ', num2str(best_nk)]);
disp(['Best validation error (MSE): ', num2str(best_val_error)]);

% Display the best model
disp('Best model:');
disp(best_model);

% Plot the actual output and predicted output
figure;
t = 1:length(output3);
plot(t, output3, 'b--', 'LineWidth', 1.5);
hold on;
plot(t, best_y_pred, 'r', 'LineWidth', 1.5);
hold off;
xlabel('Time');
ylabel('Output');
legend('Actual Output', 'Predicted Output');
title('Comparison of Actual and Predicted Output');

best_model

% Extract zeros and poles from the estimated ARX model
zeros_est = zero(best_model);
poles_est = pole(best_model);

% Display the zero locations and poles
disp('Estimated Model:');
disp('Zeros:');
disp(zeros_est);
disp('Poles:');
disp(poles_est);

% Plot the zero-pole map
figure;
zplane(zeros_est, poles_est);
title('Zero-Pole Map');

e = u.' - best_y_pred;
corr_eu = corrcoef(e, u.');
corr_eu = corr_eu(1, 2); % Extract the correlation coefficient from the matrix

% Calculate the correlation between e and y
corr_ey = corrcoef(e, best_y_pred);
corr_ey = corr_ey(1, 2); % Extract the correlation coefficient from the matrix

% Display the correlation coefficients
disp('Correlation between e and u:');
disp(corr_eu);
disp('Correlation between e and y:');
disp(corr_ey);

corr_eu = xcorr(e, u.');

% Calculate the correlation between e and y
corr_ey = xcorr(e, best_y_pred);

% Plot the correlation results
lags = -(length(e)-1):(length(e)-1);
figure;
subplot(2, 1, 1);
stem(lags, corr_eu);
title('Correlation between e and u');
xlabel('Lag');
ylabel('Correlation');
subplot(2, 1, 2);
stem(lags, corr_ey);
title('Correlation between e and y');
xlabel('Lag');
ylabel('Correlation');
