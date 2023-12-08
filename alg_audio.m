% Load audio file
audioFile = 'audio.mp3';

% Read audio file
[x, fs] = audioread(audioFile);
%sound(x,fs);

% Convert stereo to mono
x = x(:, 1);

% add gaussian noise
noiseLevel = 0.05;
noise = noiseLevel * randn(size(x));
noisySignal = x + noise;

% Apply LMS algorithm
mu_lms = 0.01; % Adjustable param
order_lms = 32; % Adjustable param
[weights_lms, output_lms] = lms_algorithm(noisySignal, x, mu_lms, order_lms);

% Apply RLS algorithm
lambda_rls = 0.99;      
order_rls = 32;            

[weights_rls, output_rls] = rls_algorithm(noisySignal, x, lambda_rls, order_rls);
coeff_rls_matlab = dsp.RLSFilter('ForgettingFactor', lambda_rls, 'Length', order_rls);
[output_rls_matlab, error_rls_matlab] = step(coeff_rls_matlab, x, noisySignal);

% Plot the results
figure;

subplot(4, 1, 1);
plot(x);
title('Original Signal');

subplot(4, 1, 2);
plot(noisySignal);
title('Noisy Signal');

subplot(4, 1, 3);
plot(output_lms);
title('LMS Signal');

subplot(4, 1, 4);
plot(output_rls);
title('RLS Signal');


% Play the original, noisy, and filtered signals
sound(x, fs);
pause(3);  
sound(noisySignal, fs);
pause(3);  
sound(output_lms, fs);
pause(3); 
sound(output_rls, fs);
pause(3);
sound(output_rls_matlab,fs);





% LMS Algorithm
function [weights, output] = lms_algorithm(input, desired, mu, order)
    % input: Input signal
    % desired: Desired signal
    % mu: Step size
    % order: Filter order
    
    N = length(input);
    weights = zeros(order, 1);
    output = zeros(N, 1);

    for i = order:N
        x = input(i:-1:i-order+1);
        y_hat = weights' * x;
        error = desired(i) - y_hat;
        weights = weights + mu * error * x;
        output(i) = y_hat;
    end
end


% RLS Algorithm
function [weights, output] = rls_algorithm(input, desired, lambda, order)
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
