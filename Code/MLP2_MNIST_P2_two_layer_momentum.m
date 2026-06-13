(* ::Package:: *)

% MLP2_MNIST _P2 _two _layer _momentum . m
% 2-hidden-layer BPNN with ReLU, softmax, cross-entropy, and momentum
clear; close all
load MNIST_train

% Data
x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

% Netk Arch 
input_size = 784;    % input size
hidden1_size = 300;  % h = 2
hidden2_size = 150;
output_size = 10;

% HE init (optimal for ReLU)
wh1 = randn(input_size, hidden1_size) * sqrt(2/input_size);
wh2 = randn(hidden1_size, hidden2_size) * sqrt(2/hidden1_size);
wo = randn(hidden2_size, output_size) * sqrt(2/hidden2_size);

% eta and momntm parameters
int = 1000;
eta = 0.1;
momentum = 0.9;

% Momentm Velocities
velocity_wo = zeros(size(wo));
velocity_wh2 = zeros(size(wh2));
velocity_wh1 = zeros(size(wh1));

ee = zeros(1, int);

fprintf('Training two-hidden-layer ReLU network with momentum...\n');

for k = 1:int
    % Fwd propagation
    vh1 = wh1' * x;
    yh1 = max(0, vh1);
    
    vh2 = wh2' * yh1;
    yh2 = max(0, vh2);
    
    vo = wo' * yh2;
    vo = vo - max(vo, [], 1);
    evo = exp(vo);
    yo = evo ./ (ones(output_size,1) * sum(evo, 1));
    
    % here is the Loss
    loss = -mean(sum(d .* log(yo + 1e-12), 1));
    ee(k) = loss;
    
    % Backprop
    delta_o = (d - yo) / 60000;
    delta_h2 = (wo * delta_o) .* (vh2 > 0);
    delta_h1 = (wh2 * delta_h2) .* (vh1 > 0);
    
    % updates
    velocity_wo = momentum * velocity_wo + eta * yh2 * delta_o';
    velocity_wh2 = momentum * velocity_wh2 + eta * yh1 * delta_h2';
    velocity_wh1 = momentum * velocity_wh1 + eta * x * delta_h1';
    
    wo = wo + velocity_wo;
    wh2 = wh2 + velocity_wh2;
    wh1 = wh1 + velocity_wh1;
    
    
    if mod(k, 100) == 0
        fprintf('Epoch % d, loss = % .4f \n', k, loss);
    end
end

% Figure the results
figure('Position', [100, 100, 800, 400]);

% Plot loss convergence
plot(1:length(ee), ee, 'b-', 'LineWidth', 2);
xlabel('Epoch', 'FontSize', 12);
ylabel('Cross-Entropy Loss', 'FontSize', 12);
title('Two-Hidden-Layer ReLU with Momentum - Loss Convergence', 'FontSize', 14);
grid on;

% Add text annotation with final accuracy (will be computed after)
hold on;
x_end = length(ee);
y_end = ee(end);
plot(x_end, y_end, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
text(x_end*0.7, y_end*3, sprintf('Final Loss: % .4f', y_end), 'FontSize', 11);

% Save the figure
saveas(gcf, 'relu_two _layer _momentum . png');
fprintf('Figure saved as relu_two _layer _momentum . png\n');

% all three subplots:
figure('Position', [100, 100, 800, 600]);

subplot(3,1,1);
plot(ee);
xlabel('Epoch');
ylabel('Loss');
title('Loss Convergence');
grid on;

subplot(3,1,2);

semilogy(ee);
xlabel('Epoch');
ylabel('Log Loss');
title('Loss (log scale)');
grid on;

subplot(3,1,3);
% Plot learning 
% show final epochs
plot(800:1000, ee(800:1000));
xlabel('Epoch');
ylabel('Loss');
title('Final 200 Epochs');
grid on;

saveas(gcf, 'relu_two _layer _momentum _detailed . png');

% Training Accur
yh1_train = max(0, wh1' * x);
yh2_train = max(0, wh2' * yh1_train);
vo_train = wo' * yh2_train;
vo_train = vo_train - max(vo_train, [], 1);
evo_train = exp(vo_train);
yo_train = evo_train ./ (ones(output_size,1) * sum(evo_train, 1));
[~, pred_train] = max(yo_train);
accuracy_train = sum (pred_train == (labelstrain + 1)') / 60000;

% Test Accur
load MNIST_test
xtest = reshape (images, [28*28 10000]) / 255;
yh1_test = max(0, wh1' * xtest);
yh2_test = max(0, wh2' * yh1_test);
vo_test = wo' * yh2_test;
vo_test = vo_test - max(vo_test, [], 1);
evo_test = exp(vo_test);
yo_test = evo_test ./ (ones(output_size,1) * sum(evo_test, 1));
[~, pred_test] = max(yo_test);
accuracy_test = sum (pred_test == (labels + 1)') / 10000;

fprintf('\n=== TWO-LAYER RELU RESULTS ===\n');
fprintf('Training accuracy: % .4f (% .2f %%)\n', accuracy_train, accuracy_train*100);
fprintf('Test accuracy: % .4f (% .2f %%)\n', accuracy_test, accuracy_test*100);
