% printer_test = 'printer_test';
% printer_train = 'printer_train -1';
% RGB_source_test = 'RGB_source_test';
% RGB_source_train = 'RGB_source_train';
% 
% printer_test = readmatrix(printer_test);
% printer_train = readmatrix(printer_train);
% RGB_source_test = readmatrix(RGB_source_test);
% RGB_source_train = readmatrix(RGB_source_train);
% 
% 


printer_test = 'printer_test.csv';
printer_train = 'printer_train -1.csv';
RGB_source_test = 'RGB_source_test.csv';
RGB_source_train = 'RGB_source_train.csv';
printer_test_data = readmatrix(printer_test);
printer_train_data = readmatrix(printer_train);
RGB_source_test_data = readmatrix(RGB_source_test);
RGB_source_train_data = readmatrix(RGB_source_train);
x_train = RGB_source_train_data(:, 1:3)';  % Input features (RGB values) for training
z_train = printer_train_data';              % Target labels for training
x_test = RGB_source_test_data(:, 1:3)';    % Input features (RGB values) for testing
z_test = printer_test_data';                % Target labels for testing