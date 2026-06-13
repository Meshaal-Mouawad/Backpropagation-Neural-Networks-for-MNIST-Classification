(* ::Package:: *)

% MLP_MNIST _P2 _two _layer _momentum . m
% Two-hidden-layer BPNN with sigmoid activation and momentum
clear; close all
load MNIST_train

% Data prep
fprintf('Preparing data...\n');
x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

% sizes
input_size = 784;
hidden1_size = 300;
hidden2_size = 150;
output_size = 10;


fprintf('Initializing weights...\n');
% add small random values to prevent saturation
wh1 = randn(input_size, hidden1_size) * 0.1;
wh2 = randn(hidden1_size, hidden2_size) * 0.1;
wo = randn(hidden2_size, output_size) * 0.1;

% Prams
sigma = 1.0;      % Sigmoid 
int = 1000;       % # of epochs
eta = 0.1;        % Learning rate
momentum = 0.9;   % Momentum 

% Vs
velocity_wo = zeros(size(wo));
velocity_wh2 = zeros(size(wh2));
velocity_wh1 = zeros(size(wh1));

ee = zeros(1, int);
grad_norms = zeros(1, int);

fprintf('Training two-hidden-layer network with momentum...\n');
fprintf('Epoch\tError\t\tGrad Norm \n');

for k = 1:int
    % Forward propagation
    % hl 1
    vh1 = wh1' * x;
    yh1 = 1 ./ (1 + exp(-sigma * vh1));
    
    % hl 2
    vh2 = wh2' * yh1;
    yh2 = 1 ./ (1 + exp(-sigma * vh2));
    
    % Output l
    vo = wo' * yh2;
    yo = 1 ./ (1 + exp(-sigma * vo));
    
    % Error
    e = d - yo;
    ee(k) = norm(e);
    
    % Back-propagation
    % Output layer delta
    delta_o = e .* yo .* (1 - yo);
    
    % hl 2 delta
    delta_h2 = (wo * delta_o) .* yh2 .* (1 - yh2);
    
    % hl 1 delta
    delta_h1 = (wh2 * delta_h2) .* yh1 .* (1 - yh1);
    
    % gradients
    dwo_raw = yh2 * delta_o';
    dwh2_raw = yh1 * delta_h2';
    dwh1_raw = x * delta_h1';
    
    grad_norms(k) = norm(dwo_raw);
    
    % Momtum updates
    velocity_wo = momentum * velocity_wo + eta * dwo_raw;
    velocity_wh2 = momentum * velocity_wh2 + eta * dwh2_raw;
    velocity_wh1 = momentum * velocity_wh1 + eta * dwh1_raw;
    
    wo = wo + velocity_wo;
    wh2 = wh2 + velocity_wh2;
    wh1 = wh1 + velocity_wh1;
    
    % Progress
    if mod(k, 100) == 0
        fprintf('% d\t% .4f\t% .2e \n', k, ee(k), grad_norms(k));
    end
end

% plot fig
figure('Position', [100, 100, 800, 600]);

subplot(2,1,1);
plot(ee);
xlabel('Epoch');
ylabel('Error');
title('Two-Hidden-Layer Sigmoid Network with Momentum - Error');
grid on;

subplot(2,1,2);
plot(grad_norms);
xlabel('Epoch');
ylabel('Gradient Norm');
title('Gradient Magnitude');
grid on;

saveas(gcf, 'two_layer _sigmoid _results . png');

% Training Accu
yh1_train = 1 ./ (1 + exp(-sigma * (wh1' * x)));
yh2_train = 1 ./ (1 + exp(-sigma * (wh2' * yh1_train)));
yo_train = 1 ./ (1 + exp(-sigma * (wo' * yh2_train)));
[~, pred_train] = max(yo_train);
accuracy_train = sum (pred_train == (labelstrain + 1)') / 60000;

% Test Accu
load MNIST_test
xtest = reshape (images, [28*28 10000]) / 255;
yh1_test = 1 ./ (1 + exp(-sigma * (wh1' * xtest)));
yh2_test = 1 ./ (1 + exp(-sigma * (wh2' * yh1_test)));
yo_test = 1 ./ (1 + exp(-sigma * (wo' * yh2_test)));
[~, pred_test] = max(yo_test);
accuracy_test = sum (pred_test == (labels + 1)') / 10000;

fprintf('\n=== TWO-LAYER SIGMOID RESULTS ===\n');
fprintf('Training accuracy: % .4f (% .2f %%)\n', accuracy_train, accuracy_train*100);
fprintf('Test accuracy: % .4f (% .2f %%)\n', accuracy_test, accuracy_test*100);
