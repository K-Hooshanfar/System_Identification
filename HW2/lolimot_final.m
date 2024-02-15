function [y_pred] = lolimot_final(epoch, x_train, z_train, x_test, z_test)
    x = x_train;
    y = z_train;
    d = size(x_train);
    d = d(2); % number of train datas
    partx = cell(1, 1); % datas of regions
    partx{1} = x;
    c = [0, 0]; % centers of regions, defining the center of the whole data set
    sig = cell(1, 1); % variances of regions
    sig{1} = (400/9) * eye(2); % defining the variances of the whole data set
    lx = [20, 20]; % length of regions, defined for the whole data set
    phi = zeros(1, d); % activation function
    a = zeros(1, d); % a variable that is used later

    % defining activation function for the whole data set
    for j = 1:d
        phi(1, j) = exp(-0.5 * x_train(:, j)' * (eye(2) / sig{1}) * x_train(:, j));
    end

    p = phi; % sum of all activation functions
    X = [ones(d, 1), x_train']; % regressor's matrix
    Q = zeros(d); % weight matrix
    for i = 1:d
        Q(i, i) = 1;
    end

    theta = pinv(X' * Q * X) * X' * Q * y'; % model linear parameters
    ys = X * theta; % estimated output for the first iteration
    mse_test = zeros(1, epoch); % mean squared error of test data estimation
    mse = zeros(1, epoch); % mean squared error of train data estimation

    for M = 1:epoch
        % local costs
        I_loc = zeros(1, M);

        % computing the local costs and choosing the one that is bigger
        for i = 1:M
            for j = 1:d
                a(j) = ((y(j) - ys(j))^2) * phi(i, j) / p(j);
            end
            I_loc(i) = sum(a);
            a = zeros(1, d);
        end

        [~, n_loc] = max(I_loc);
        x_loc = partx{n_loc}; % choosing a part that has bigger cost
        c_loc = c(n_loc, :);

        % splitting this part into two left and right sides
        r = find(x_loc(1, :) > c_loc(1));
        l = find(x_loc(1, :) < c_loc(1));
        num_r = size(r);
        num_r = num_r(2); % number of datas in the right half
        num_l = size(l);
        num_l = num_l(2); % number of datas in the left half
        x_loc_r = zeros(2, num_r); % datas in the right side
        x_loc_l = zeros(2, num_l); % datas in the left side

        for i = 1:num_r
            x_loc_r(:, i) = x_loc(:, r(i));
        end

        for i = 1:num_l
            x_loc_l(:, i) = x_loc(:, l(i));
        end

        % splitting this part into two up and down sides
        u = find(x_loc(2, :) > c_loc(2));
        do = find(x_loc(2, :) < c_loc(2));
        num_u = size(u);
        num_u = num_u(2); % number of datas in the upper side
        num_d = size(do);
        num_d = num_d(2); % number of datas in the lower side
        x_loc_d = zeros(2, num_d); % datas in the upper side
        x_loc_u = zeros(2, num_u); % datas in the lower side

        for i = 1:num_u
            x_loc_u(:, i) = x_loc(:, u(i));
        end

        for i = 1:num_d
            x_loc_d(:, i) = x_loc(:, do(i));
        end

        % computing global costs for two different ways of splitting the selected region
        I_lr = globcost(x_train, n_loc, sig, c, lx, phi, y, 1);
        I_ud = globcost(x_train, n_loc, sig, c, lx, phi, y, 2);

        % defining new variables for new regions
        sig2 = cell(1, M + 1);
        c2 = zeros(M + 1, 2);
        partx2 = cell(1, M + 1);
        lx2 = zeros(M + 1, 2);
        phi2 = zeros(M + 1, d);

        % putting the unselected regions' information in these new variables
        for i = 1:M
            if i ~= n_loc
                sig2{i} = sig{i};
                c2(i, :) = c(i, :);
                partx2{i} = partx{i};
                lx2(i, :) = lx(i, :);
                phi2(i, :) = phi(i, :);
            end
        end

        % choosing the two new regions that their corresponding global cost is lower
        % and putting the information of these two parts in the new variables
        if I_lr < I_ud
            sig2{n_loc} = [1/4, 0; 0, 1] * sig{n_loc};
            sig2{M + 1} = [1/4, 0; 0, 1] * sig{n_loc};
            c2(n_loc, :) = [(lx(n_loc, 1) / 4) + c(n_loc, 1), c(n_loc, 2)];
            c2(M + 1, :) = [-(lx(n_loc, 1) / 4) + c(n_loc, 1), c(n_loc, 2)];
            partx2{n_loc} = x_loc_r;
            partx2{M + 1} = x_loc_l;
            lx2(n_loc, 1) = 0.5 * lx(n_loc, 1);
            lx2(n_loc, 2) = lx(n_loc, 2);
            lx2(M + 1, 1) = 0.5 * lx(n_loc, 1);
            lx2(M + 1, 2) = lx(n_loc, 2);

            for j = 1:d
                phi2(n_loc, j) = exp(-0.5 * (x_train(:, j) - c2(n_loc, :)')' * (eye(2) / sig2{n_loc}) * (x_train(:, j) - c2(n_loc, :)'));
                phi2(M + 1, j) = exp(-0.5 * (x_train(:, j) - c2(M + 1, :)')' * (eye(2) / sig2{M+ 1}) * (x_train(:, j) - c2(M + 1, :)'));
            end
        else
            sig2{n_loc} = [1, 0; 0, 1/4] * sig{n_loc};
            sig2{M + 1} = [1, 0; 0, 1/4] * sig{n_loc};
            c2(n_loc, :) = [c(n_loc, 1), (1/4) * lx(n_loc, 2) + c(n_loc, 2)];
            c2(M + 1, :) = [c(n_loc, 1), -(1/4) * lx(n_loc, 2) + c(n_loc, 2)];
            partx2{n_loc} = x_loc_u;
            partx2{M + 1} = x_loc_d;
            lx2(n_loc, 1) = lx(n_loc, 1);
            lx2(n_loc, 2) = 0.5 * lx(n_loc, 2);
            lx2(M + 1, 1) = lx(n_loc, 1);
            lx2(M + 1, 2) = 0.5 * lx(n_loc, 2);

            for j = 1:d
                phi2(n_loc, j) = exp(-0.5 * (x_train(:, j) - c2(n_loc, :)')' * (eye(2) / sig2{n_loc}) * (x_train(:, j) - c2(n_loc, :)'));
                phi2(M + 1, j) = exp(-0.5 * (x_train(:, j) - c2(M + 1, :)')' * (eye(2) / sig2{M + 1}) * (x_train(:, j) - c2(M + 1, :)'));
            end
        end

        % changing the name of new variables so they could be used in the next iteration
        phi = zeros(M + 1, d);
        sig = cell(1, M + 1);
        c = zeros(M + 1, 2);
        partx = cell(1, M + 1);
        lx = zeros(M + 1, 2);

        for i = 1:M + 1
            phi(i, :) = phi2(i, :);
            sig{i} = sig2{i};
            c(i, :) = c2(i, :);
            partx{i} = partx2{i};
            lx(i, :) = lx2(i, :);
        end

        % computing the estimated output
        theta = zeros(3, M + 1);
        ys = zeros(1, d);
        p = sum(phi);

        for i = 1:M + 1
            for j = 1:d
                Q(j, j) = phi(i, j) / p(j);
            end

            theta(:, i) = pinv(X' * Q * X) * X' * Q * y';

            for j = 1:d
                ys(j) = ys(j) + X(j, :) * theta(:, i) * phi(i, j) / p(j);
            end
        end

        y_pred = LLNFtest(theta, c, sig, x_test); % computing estimated output for test data
        mse(M) = (1 / d) * sum((y - ys).^2);
        mse_test(M) = (1 / 71) * sum((z_test - y_pred).^2);
    end

        % plotting MSE 
    figure(1)
    plot(mse);
    hold on
    plot(mse_test, 'r');
    legend('Train Data', 'Test Data');
    title('Mean Squared Error for Train and Test Data');
    xlabel('Iterations');
    ylabel('MSE');
end
