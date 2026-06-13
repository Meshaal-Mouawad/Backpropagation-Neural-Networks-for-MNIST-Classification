(* ::Package:: *)

% MLP_MNIST _P2 _with _momentum _ 5.m
% One-hidden-layer BPNN with sigmoid activation and momentum
clear; close all
load MNIST_train

% Data preparation
x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

wo = randn(300, 10) * 0.1;  % Small random values
wh = randn(28*28, 300) * 0.1;

% Params
sigma = 1.0;      % Sigmoid slope
int = 1000;
eta = 0.1;        % Higher learning rate
momentum = 0.9;

% int velocities
velocity_wo = zeros(size(wo));
velocity_wh = zeros(size(wh));

% tracking
ee = zeros(1, int);
grad_norms = zeros(1, int);

fprintf('Training one-hidden-layer network with momentum...\n');
fprintf('Epoch\tError\t\tGrad Norm \n');

for k = 1:int
    % Forward propagation
    vh = wh' * x;
    yh = 1 ./ (1 + exp(-sigma * vh));
    
    vo = wo' * yh;
    yo = 1 ./ (1 + exp(-sigma * vo));
    
    % Error
    e = d - yo;
    ee(k) = norm(e);
    
    % Gradients
    delta_o = e .* yo .* (1 - yo);
    delta_h = (wo * delta_o) .* yh .* (1 - yh);
    
    dwo_raw = yh * delta_o';
    dwh_raw = x * delta_h';
    
    grad_norms(k) = norm(dwo_raw);
    
    % Momentum updates
    velocity_wo = momentum * velocity_wo + eta * dwo_raw;
    velocity_wh = momentum * velocity_wh + eta * dwh_raw;
    
    wo = wo + velocity_wo;
    wh = wh + velocity_wh;
    
    % Progress
    if mod(k, 100) == 0
        fprintf('% d\t% .4f\t% .2e \n', k, ee(k), grad_norms(k));
    end
end

% Plot convergence
% fig 5, last update
figure;
subplot(2,1,1);
plot(ee);
xlabel('Epoch');
ylabel('Error');
title('One-Hidden-Layer with Momentum - Error Convergence');
grid on;

subplot(2,1,2);
plot(grad_norms);
xlabel('Epoch');
ylabel('Gradient Norm');
title('Gradient Magnitude (should be non-zero)');
grid on;

% Save figure
saveas(gcf, 'one_layer _momentum _results . png');

% Training accu
yh_train = 1 ./ (1 + exp(-sigma * (wh' * x)));
yo_train = 1 ./ (1 + exp(-sigma * (wo' * yh_train)));
[~, pred_train] = max(yo_train);
accuracy_train = sum (pred_train == (labelstrain + 1)') / 60000;

% Test accu
load MNIST_test
xtest = reshape (images, [28*28 10000]) / 255;
yh_test = 1 ./ (1 + exp(-sigma * (wh' * xtest)));
yo_test = 1 ./ (1 + exp(-sigma * (wo' * yh_test)));
[~, pred_test] = max(yo_test);
accuracy_test = sum (pred_test == (labels + 1)') / 10000;

fprintf('\n=== RESULTS WITH MOMENTUM ===\n');
fprintf('Training accuracy: % .4f (% .2f %%)\n', accuracy_train, accuracy_train*100);
fprintf('Test accuracy: % .4f (% .2f %%)\n', accuracy_test, accuracy_test*100);
