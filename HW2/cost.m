function I = cost(x_train, n_loc, sig, c, lx, phi, y, a) 
% we define this function the thing that was said in the book


    % Constructing the center and variances for the two situations:
    % a=1 then the part is split vertically
    % a=2 then the part is split horizontally
    if a == 1
        c1 = c(n_loc, :) + (1/4) * [lx(1,1), 0];
        c2 = c(n_loc, :) - (1/4) * [lx(1,1), 0];
        sig2 = [1/4, 0; 0, 1] * sig{n_loc};
    else
        c1 = c(n_loc, :) + (1/4) * [0, lx(1,2)];
        c2 = c(n_loc, :) - (1/4) * [0, lx(1,2)];
        sig2 = [1, 0; 0, 1/4] * sig{n_loc};
    end
    
    d = size(x_train, 2); % size of train data
    M = size(c, 1); % number of parts
    phi2 = zeros(M+1, d); % the new activation functions
    X = [ones(d, 1), x_train']; % regressors matrix
    
    % Constructing new activation function
    for i = 1:M
        if i ~= n_loc
            phi2(i, :) = phi(i, :);
        end
    end
    
    % Computing activation functions for two new parts that take the place of the selected part
    for j = 1:d
        phi2(n_loc, j) = exp(-0.5 * (x_train(:,j) - c1')' * (eye(2) / sig2) * (x_train(:,j) - c1'));
        phi2(M+1, j) = exp(-0.5 * (x_train(:,j) - c2')' * (eye(2) / sig2) * (x_train(:,j) - c2'));
    end
    
    % Computing the global cost
    theta = zeros(3, M+1);
    ys = zeros(1, d);
    Q = zeros(d);
    p = sum(phi2); % sum of activation functions
    
    for i = 1:M+1
        for j = 1:d
            Q(j, j) = phi2(i, j) / p(j); % weight matrix
        end
        theta(:, i) = pinv(X' * Q * X) * X' * Q * y';
        
        % Computing output estimation
        for j = 1:d
            ys(j) = ys(j) + X(j, :) * theta(:, i) * phi2(i, j) / p(j);
        end
    end
    
    I = sum((y - ys).^2);
end
