(* ::Package:: *)

% MLP2_MNIST _P2 _with _momentum . m
% One-hidden-layer BPNN with ReLU, softmax, cross-entropy, and momentum
clear; close all
load MNIST_train

x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

hidden_size = 300;
output_size = 10;

% Xavier
wo = randn (hidden_size, output_size) / sqrt(hidden_size);
wh = randn (28*28, hidden_size) / sqrt(28*28);


int = 1000;
eta = 0.1;        % Learnin rate
momentum = 0.9;   % Momentuo coef

% Velocities
velocity_wo = zeros(size(wo));
velocity_wh = zeros(size(wh));

ee = zeros(1, int);
grad_norms = zeros(1, int);

fprintf('Training one-hidden-layer ReLU network with momentum...\n');
fprintf('Epoch\tLoss\t\tGrad Norm \n');

for k = 1:int
    % Fwd propagation
    vh = wh' * x;
    yh = max(0, vh);  % Relu
    
    vo = wo' * yh;
    vo = vo - max(vo, [], 1);  
    evo = exp(vo);
    yo = evo ./ (ones(output_size,1) * sum(evo, 1));
    
    % loss 
    loss = -mean(sum(d .* log(yo + 1e-12), 1));
    ee(k) = loss;
    
    % Backpropagation 
    delta_o = (d - yo) / 60000;  % Softmax dervative
    delta_h = (wo * delta_o) .* (vh > 0);  % ReLU derivative
    
    dwo_raw = yh * delta_o';
    dwh_raw = x * delta_h';
    
    grad_norms(k) = norm(dwo_raw);
    
    % Momntm updates
    velocity_wo = momentum * velocity_wo + eta * dwo_raw;
    velocity_wh = momentum * velocity_wh + eta * dwh_raw;
    
    wo = wo + velocity_wo;
    wh = wh + velocity_wh;
    
    
    if mod(k, 100) == 0
        fprintf('% d\t% .4f\t% .2e \n', k, ee(k), grad_norms(k));
    end
end

% Compression Figure
figure('Position', [100, 100, 800, 600]);

subplot(2,1,1);
plot(ee);
xlabel('Epoch');
ylabel('Cross-Entropy Loss');
title('One-Hidden-Layer ReLU with Momentum - Loss Convergence');
grid on;

subplot(2,1,2);
plot(grad_norms);
xlabel('Epoch');
ylabel('Gradient Norm');
title('Gradient Magnitude');
grid on;

saveas(gcf, 'relu_one _layer _momentum . png');

% Training Accr
yh_train = max(0, wh' * x);
vo_train = wo' * yh_train;
vo_train = vo_train - max(vo_train, [], 1);
evo_train = exp(vo_train);
yo_train = evo_train ./ (ones(output_size,1) * sum(evo_train, 1));
[~, pred_train] = max(yo_train);
accuracy_train = sum (pred_train == (labelstrain + 1)') / 60000;

% Testing Accr
load MNIST_test
xtest = reshape (images, [28*28 10000]) / 255;
yh_test = max(0, wh' * xtest);
vo_test = wo' * yh_test;
vo_test = vo_test - max(vo_test, [], 1);
evo_test = exp(vo_test);
yo_test = evo_test ./ (ones(output_size,1) * sum(evo_test, 1));
[~, pred_test] = max(yo_test);
accuracy_test = sum (pred_test == (labels + 1)') / 10000;

fprintf('\n=== ONE-LAYER RELU RESULTS ===\n');
fprintf('Training accuracy: % .4f (% .2f %%)\n', accuracy_train, accuracy_train*100);
fprintf('Test accuracy: % .4f (% .2f %%)\n', accuracy_test, accuracy_test*100);
