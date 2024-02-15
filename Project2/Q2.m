tbl = readtable('Battery.xlsx', 'PreserveVariableNames', true);
% V = tbl(:, 'V(mV)');
% I = tbl(:, 'I(mA)');
validation_size = 500;
V = tbl{:, 'V(mV)'};
I = tbl{:, 'I(mA)'};

% Shuffle and split for V
column1_shuffled = V(randperm(numel(V)));
u_val = column1_shuffled(end-validation_size+1:end);
u_train = column1_shuffled(1:end-validation_size);
% Shuffle and split for I
column2_shuffled = I(randperm(numel(I)));
y_val = column2_shuffled(end-validation_size+1:end);
y_train = column2_shuffled(1:end-validation_size);

na_values = [1, 2, 3 ,4,5,6];  % List of 'na' values to test
nc_values = [1, 2, 3,4,6,5];  % List of 'na' values to test
nb_values = [1, 2, 3 ,4,6,5];  % List of 'nb' values to test
nf_values = [1, 2, 3 ,4,6,5];  % List of 'na' values to test
nd_values = [1, 2, 3,4,6,5];  % List of 'na' values to test
nk_values = [0, 1, 2, 3,4,6,5];  % List of 'nk' values to test
best_na = 0;            % Variable to store the best 'na' value
best_nb = 0;            % Variable to store the best 'nb' value
best_nk = 0;            % Variable to store the best 'nk' value
best_nc = 0;            % Variable to store the best 'nk' value
best_nf = 0;            % Variable to store the best 'nk' value
best_nd = 0;            % Variable to store the best 'nk' value
best_val_error = Inf;   % Variable to store the best validation error
best_model = [];        % Variable to store the best model

best_y_pred = []; 
% % Loop over each combination of 'na', 'nb', 'nc', and 'nk' values
% for na = na_values
%     for nb = nb_values
%         for nc = nc_values
%             for nk = nk_values
%                 % ARMAX Model Identification
%                 sys = armax(iddata(y_train, u_train), [na, nb, nc, nk]);
%                 y_val_pred = compare(sys, iddata(y_val, u_val)); 
%                 val_error = mean((y_val_pred.OutputData - y_val).^2);
%                 if val_error < best_val_error
%                     best_val_error = val_error;
%                     best_na = na;
%                     best_nb = nb;
%                     best_nc = nc;
%                     best_nk = nk;
%                     best_model = sys;  % Store the best model
%                      best_y_pred = y_val_pred.OutputData;
%                 end
%             end
%         end
%     end
% end
% 
% % Display the best parameter values and validation error
% disp(['Best na value: ', num2str(best_na)]);
% disp(['Best nb value: ', num2str(best_nb)]);
% disp(['Best nc value: ', num2str(best_nc)]);
% disp(['Best nk value: ', num2str(best_nk)]);
% disp(['Best validation error (MSE): ', num2str(best_val_error)]);



% % Loop over each combination of 'na', 'nb', and 'nk' values
% for na = na_values
%     for nb = nb_values
%         for nk = nk_values
%             % ARX Model Identification
%             sys = arx(iddata(y_train, u_train), [na, nb, nk]);
%             y_val_pred = compare(sys, iddata(y_val, u_val)); 
%             val_error = mean((y_val_pred.OutputData - y_val).^2);
%             if val_error < best_val_error
%                 best_val_error = val_error;
%                 best_na = na;
%                 best_nb = nb;
%                 best_nk = nk;
%                 best_model = sys;  % Store the best model
%                 best_y_pred = y_val_pred.OutputData;
%             end
%         end
%     end
% end
% 
% % Display the best parameter values and validation error
% disp(['Best na value: ', num2str(best_na)]);
% disp(['Best nb value: ', num2str(best_nb)]);
% disp(['Best nk value: ', num2str(best_nk)]);
% disp(['Best validation error (MSE): ', num2str(best_val_error)]);


% 
% % Loop over each combination of 'nb', 'nc', 'nd', 'nf', and 'nk' values
% for nb = nb_values
%     for nc = nc_values
%         for nd = nd_values
%             for nf = nf_values
%                 for nk = nk_values
%                     % Box-Jenkins Model Identification
%                     orders = [0 nb nc nd nf nk];
%                     sys = polyest(iddata(y_train, u_train), orders);
%                     y_val_pred = compare(sys, iddata(y_val, u_val)); 
%                     val_error = mean((y_val_pred.OutputData - y_val).^2);
%                     if val_error < best_val_error
%                         best_val_error = val_error;
%                         best_nb = nb;
%                         best_nc = nc;
%                         best_nd = nd;
%                         best_nf = nf;
%                         best_nk = nk;
%                         best_model = sys;  % Store the best model
%                         best_y_pred = y_val_pred.OutputData;
%                     end
%                 end
%             end
%         end
%     end
% end
% 
% % Display the best parameter values and validation error
% disp(['Best nb value: ', num2str(best_nb)]);
% disp(['Best nc value: ', num2str(best_nc)]);
% disp(['Best nd value: ', num2str(best_nd)]);
% disp(['Best nf value: ', num2str(best_nf)]);
% disp(['Best nk value: ', num2str(best_nk)]);
% disp(['Best validation error (MSE): ', num2str(best_val_error)]);
% 
% % Display the best model
% disp('Best model:');
% disp(best_model);


% % Loop over each combination of 'nb', 'nf', and 'nk' values
% for nb = nb_values
%     for nf = nf_values
%         for nk = nk_values
%             % Output-Error Model Identification
%             sys = oe(iddata(y_train, u_train), [nb, nf, nk]);
%             y_val_pred = compare(sys, iddata(y_val, u_val));
%             val_error = mean((y_val_pred.OutputData - y_val).^2);
%             if val_error < best_val_error
%                 best_val_error = val_error;
%                 best_nb = nb;
%                 best_nf = nf;
%                 best_nk = nk;
%                 best_model = sys;  % Store the best model
%                 best_y_pred = y_val_pred.OutputData;
%             end
%         end
%     end
% end
% 
% % Display the best parameter values and validation error
% disp(['Best nb value: ', num2str(best_nb)]);
% disp(['Best nf value: ', num2str(best_nf)]);
% disp(['Best nk value: ', num2str(best_nk)]);
% disp(['Best validation error (MSE): ', num2str(best_val_error)]);
% 
% % Display the best model
% disp('Best model:');
% disp(best_model);



for na = na_values
    for nb = nb_values
        for nd = nd_values
            for nk = nk_values
                % PEM Model Identification
                model = pem(iddata(y_train, u_train), 'na', na, 'nb', nb, 'nd', nd, 'nk', nk);
                y_val_pred = sim(model, iddata(y_val, u_val));
                val_error = mean((y_val_pred.OutputData - y_val).^2);
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
t = 1:length(y_val);
plot(t, y_val, 'b--', 'LineWidth', 1.5);
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














% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Load the voltage and current data from the Excel file
% data = Battery;
% voltage = data(:, 1);
% current = data(:, 2);
% 
% % Split the data into training and validation sets
% validation_samples = 500;
% train_voltage = voltage(validation_samples + 1:end);
% train_current = current(validation_samples + 1:end);
% valid_voltage = voltage(1:validation_samples);
% valid_current = current(1:validation_samples);
% 
% % Create the model structures
% order = 2;  % Set the desired order for the models
% ARX_model = arx([train_voltage, train_current], order);
% ARMAX_model = armax([train_voltage, train_current], [order, order, 0]);
% ARARX_model = ararx([train_voltage, train_current], order, order);
% BJ_model = bj([train_voltage, train_current], [order, order, 0, 0, 0]);
% 
% % Validate the models
% compare(ARX_model, valid_voltage);
% compare(ARMAX_model, valid_voltage);
% compare(ARARX_model, valid_voltage);
% compare(BJ_model, valid_voltage);
% 
% % Select the best model based on validation performance
% % You can compare the fit, prediction error, or any other metric to make the decision
% 
% % Use the selected model for prediction and repair
% predicted_voltage = forecast(BJ_model, [current, voltage], validation_samples);
% % Implement the repair process based on the predicted voltage values
% 
% % Additional analysis or visualization can be performed as needed
% 
% 


