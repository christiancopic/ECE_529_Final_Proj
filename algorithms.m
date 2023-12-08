% Generate a random input signal
rng(50);  % Set seed for reproducibility
N = 200;
input = randn(N, 1);

% Define a filter
filter = [1, -0.5, 0.2];

% Generate a desired signal by applying the filter to the input
desired = filter(1) * input + filter(2) * [0; input(1:end-1)] + filter(3) * [0; 0; input(1:end-2)];
desired_a = desired;

% Add noise
desired = desired + 0.1 * randn(N, 1);


% Apply the LMS algorithm
mu = 0.01;
order = length(filter);
[weights_lms, error_lms, output_lms] = lms_algorithm(input, desired, mu, order);

% Apply the RLS algorithm
lambda = 0.99;
[weights_rls, error_rls, output_rls] = rls_algorithm(input, desired, lambda, order);

% Compare with MATLAB DSP Toolbox
coeff_lms_matlab = dsp.LMSFilter('StepSize', mu, 'Length', order);
[output_lms_matlab, error_lms_matlab] = step(coeff_lms_matlab, desired, input);

coeff_rls_matlab = dsp.RLSFilter('ForgettingFactor', lambda, 'Length', order);
[output_rls_matlab, error_rls_matlab] = step(coeff_rls_matlab, desired, input);



% Plot the results
figure;

subplot(2, 2, 1);
plot(1:N, error_lms, 'b', 1:N, error_lms_matlab, 'r--');
title('LMS Algorithm Error');
legend('Custom Implementation', 'MATLAB Toolbox');

subplot(2, 2, 2);
stem(filter, 'b', 'LineWidth', 2);
hold on;
stem(weights_lms, 'r--', 'LineWidth', 1);
title('Filter Coefficients (LMS)');
legend('True Filter', 'Estimated (LMS)');

subplot(2, 2, 3);
plot(1:N, error_rls, 'b', 1:N, error_rls_matlab, 'r--');
title('RLS Algorithm Error');
legend('Custom Implementation', 'MATLAB Toolbox');

subplot(2, 2, 4);
stem(filter, 'b', 'LineWidth', 2);
hold on;
stem(weights_rls, 'r--', 'LineWidth', 1);
title('Filter Coefficients (RLS)');
legend('True Filter', 'Estimated (RLS)');

figure;
subplot(2,1,1);
plot(1:N, output_lms, 'b', 1:N, output_lms_matlab, 'r--', 1:N, desired_a);
title('LMS Output');
legend('Custom Implementation', 'MATLAB Toolbox', 'Desired Output');

subplot(2,1,2);
plot(1:N, output_rls, 'b', 1:N, output_rls_matlab, 'r--', 1:N, desired_a);
title('RLS Output');
legend('Custom Implementation', 'MATLAB Toolbox', 'Desired Output');



% LMS Algorithm
function [weights, error, output] = lms_algorithm(input, desired, mu, order)
    % input: Input signal
    % desired: Desired signal
    % mu: Step size
    % order: Filter order
    
    N = length(input);
    weights = zeros(order, 1);
    error = zeros(N, 1);
    output = zeros(N, 1);
    
    for i = order:N
        x = input(i:-1:i-order+1);
        y_hat = weights' * x;
        error(i) = desired(i) - y_hat;
        weights = weights + mu * error(i) * x;
        output(i) = y_hat;
    end
end

% RLS Algorithm
function [weights, error, output] = rls_algorithm(input, desired, lambda, order)
    % input: Input signal
    % desired: Desired signal
    % lambda: Forgetting factor
    % order: Filter order
    
    N = length(input);
    P = eye(order);
    weights = zeros(order, 1);
    error = zeros(N, 1);
    output = zeros(N, 1);
    
    for i = order:N
        x = input(i:-1:i-order+1);
        phi = P * x;
        k = (phi) / (lambda + x' * phi);
        error(i) = desired(i) - weights' * x;
        weights = weights + k * error(i);
        y_hat = weights' * x;
        P = (P - k * phi') / lambda;
        output(i) = y_hat;
    end
end
