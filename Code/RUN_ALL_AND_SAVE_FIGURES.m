(* ::Package:: *)

% RUN_ALL _AND _SAVE _FIGURES . m
% This script runs all your project codes and saves figures with white backgrounds

clear; close all; clc;
fprintf('========================================\n');
fprintf('REGENERATING ALL PROJECT FIGURES \n');
fprintf('========================================\n \n');

% Create a folder for the new figures
if ~exist('ProjectFigures', 'dir')
    mkdir('ProjectFigures');
end


% PART 1: SIGMOID FIGS
fprintf('\n--- PART 1: Sigmoid Network Figures ---\n');

%% (MLP_MNIST _P2 . m)
fprintf('Running original sigmoid network...\n');
clear; close all; % Fresh start
load MNIST_train

x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

wo = rand(300,10)-0.5;
wh = rand(28*28,300)-0.5; 

sigma = 0.01;
int = 1000;
eta = 0.001;

ee = zeros(1, int);
for k = 1:int
    vh = wh'*x;
    yh = 1./(1+exp(-sigma*vh));
    vo = wo'*yh;
    yo = 1./(1+exp(-sigma*vo));
    e = d - yo;
    ee(k) = norm(e);
    
    delta_o = e .*yo .*(1-yo);
    dwo = eta*yh*delta_o';
    delta_h = (wo*delta_o) .*yh .*(1-yh);
    dwh = eta*x*delta_h';
    
    wo = wo + dwo;
    wh = wh + dwh;
end

% Create figure
fig1 = figure('Position', [100, 100, 600, 400], 'Color', 'white');
plot(ee, 'b-', 'LineWidth', 1.5);
xlabel('Epoch', 'FontSize', 12);
ylabel('Error', 'FontSize', 12);
title('Original Sigmoid Network Convergence', 'FontSize', 14);
grid on;
set(gca, 'Color', 'white', 'GridColor', [0.8 0.8 0.8]);
saveas(fig1, fullfile('ProjectFigures', 'NN2 . png'));
saveas(fig1, fullfile('ProjectFigures', 'NN2 . fig'));
fprintf('  Saved: ProjectFigures/NN2 . png\n');
close(fig1);

%% Sigmoid with Mome
fprintf('\n Running sigmoid with momentum...\n');
clear; close all;
load MNIST_train

x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

wo = randn(300, 10) * 0.1;
wh = randn(28*28, 300) * 0.1;

sigma = 1.0;
int = 1000;
eta = 0.1;
momentum = 0.9;

velocity_wo = zeros(size(wo));
velocity_wh = zeros(size(wh));
ee = zeros(1, int);

for k = 1:int
    vh = wh' * x;
    yh = 1 ./ (1 + exp(-sigma * vh));
    vo = wo' * yh;
    yo = 1 ./ (1 + exp(-sigma * vo));
    e = d - yo;
    ee(k) = norm(e);
    
    delta_o = e .* yo .* (1 - yo);
    delta_h = (wo * delta_o) .* yh .* (1 - yh);
    
    dwo_raw = yh * delta_o';
    dwh_raw = x * delta_h';
    
    velocity_wo = momentum * velocity_wo + eta * dwo_raw;
    velocity_wh = momentum * velocity_wh + eta * dwh_raw;
    
    wo = wo + velocity_wo;
    wh = wh + velocity_wh;
end

fig2 = figure('Position', [100, 100, 600, 400], 'Color', 'white');
plot(ee, 'b-', 'LineWidth', 1.5);
xlabel('Epoch', 'FontSize', 12);
ylabel('Error', 'FontSize', 12);
title('Sigmoid Network with Momentum', 'FontSize', 14);
grid on;
set(gca, 'Color', 'white', 'GridColor', [0.8 0.8 0.8]);
saveas(fig2, fullfile('ProjectFigures', 'MNISTMoment . png'));
saveas(fig2, fullfile('ProjectFigures', 'MNISTMoment . fig'));
fprintf('  Saved: ProjectFigures/MNISTMoment . png\n');
close(fig2);

%% l=2 Sigmoid
fprintf('\n Running two-layer sigmoid...\n');
clear; close all;
load MNIST_train

x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

input_size = 784;
hidden1_size = 300;
hidden2_size = 150;
output_size = 10;

wh1 = randn(input_size, hidden1_size) * 0.1;
wh2 = randn(hidden1_size, hidden2_size) * 0.1;
wo = randn(hidden2_size, output_size) * 0.1;

sigma = 1.0;
int = 1000;
eta = 0.1;
momentum = 0.9;

velocity_wo = zeros(size(wo));
velocity_wh2 = zeros(size(wh2));
velocity_wh1 = zeros(size(wh1));
ee = zeros(1, int);

for k = 1:int
    vh1 = wh1' * x;
    yh1 = 1 ./ (1 + exp(-sigma * vh1));
    vh2 = wh2' * yh1;
    yh2 = 1 ./ (1 + exp(-sigma * vh2));
    vo = wo' * yh2;
    yo = 1 ./ (1 + exp(-sigma * vo));
    e = d - yo;
    ee(k) = norm(e);
    
    delta_o = e .* yo .* (1 - yo);
    delta_h2 = (wo * delta_o) .* yh2 .* (1 - yh2);
    delta_h1 = (wh2 * delta_h2) .* yh1 .* (1 - yh1);
    
    dwo_raw = yh2 * delta_o';
    dwh2_raw = yh1 * delta_h2';
    dwh1_raw = x * delta_h1';
    
    velocity_wo = momentum * velocity_wo + eta * dwo_raw;
    velocity_wh2 = momentum * velocity_wh2 + eta * dwh2_raw;
    velocity_wh1 = momentum * velocity_wh1 + eta * dwh1_raw;
    
    wo = wo + velocity_wo;
    wh2 = wh2 + velocity_wh2;
    wh1 = wh1 + velocity_wh1;
end

fig3 = figure('Position', [100, 100, 600, 400], 'Color', 'white');
plot(ee, 'b-', 'LineWidth', 1.5);
xlabel('Epoch', 'FontSize', 12);
ylabel('Error', 'FontSize', 12);
title('Two-Layer Sigmoid Network', 'FontSize', 14);
grid on;
set(gca, 'Color', 'white', 'GridColor', [0.8 0.8 0.8]);
saveas(fig3, fullfile('ProjectFigures', 'sigmoid_twolayer _convergence . png'));
saveas(fig3, fullfile('ProjectFigures', 'sigmoid_twolayer _convergence . fig'));
fprintf('  Saved: ProjectFigures/sigmoid_twolayer_convergence . png\n');
close(fig3);

=
% RELU Figs

fprintf('\n--- PART 2: ReLU Network Figures ---\n');

%% Original ReLU (MLP2_MNIST _P2 . m)
fprintf('Running original ReLU network...\n');
clear; close all;
load MNIST_train

x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

wo = randn(300,10)*sqrt(1/784);
wh = randn(28*28,300)*sqrt(1/784); 

int = 1000;
eta = 0.5;

ee = zeros(1, int);
for k = 1:int
    vh = wh'*x;
    yh = max(0, vh);
    vo = wo'*yh;
    vo = vo - max(vo,[],1);
    evo = exp(vo);
    yo = evo ./(ones(10,1)*sum(evo,1));
    e = -mean(sum(d .*log(yo + 1e-12),1));
    ee(k) = e;
    
    delta_o = (d-yo)/60000;
    dwo = eta*yh*delta_o';
    delta_h = (wo*delta_o) .*double(vh>0);
    dwh = eta*x*delta_h';
    
    wo = wo + dwo;
    wh = wh + dwh;
end

fig4 = figure('Position', [100, 100, 600, 400], 'Color', 'white');
plot(ee, 'b-', 'LineWidth', 1.5);
xlabel('Epoch', 'FontSize', 12);
ylabel('Loss', 'FontSize', 12);
title('Original ReLU Network Convergence', 'FontSize', 14);
grid on;
set(gca, 'Color', 'white', 'GridColor', [0.8 0.8 0.8]);
saveas(fig4, fullfile('ProjectFigures', 'relu_original _convergence . png'));
saveas(fig4, fullfile('ProjectFigures', 'relu_original _convergence . fig'));
fprintf('  Saved: ProjectFigures/relu_original_convergence . png\n');
close(fig4);

%% ReLU |+ Momentum
fprintf('\n Running ReLU with momentum...\n');
clear; close all;
load MNIST_train

x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

wo = randn(300,10)*sqrt(1/784);
wh = randn(28*28,300)*sqrt(1/784); 

int = 1000;
eta = 0.5;
momentum = 0.9;

velocity_wo = zeros(size(wo));
velocity_wh = zeros(size(wh));
ee = zeros(1, int);

for k = 1:int
    vh = wh'*x;
    yh = max(0, vh);
    vo = wo'*yh;
    vo = vo - max(vo,[],1);
    evo = exp(vo);
    yo = evo ./(ones(10,1)*sum(evo,1));
    e = -mean(sum(d .*log(yo + 1e-12),1));
    ee(k) = e;
    
    delta_o = (d-yo)/60000;
    delta_h = (wo*delta_o) .*double(vh>0);
    
    velocity_wo = momentum * velocity_wo + eta * yh * delta_o';
    velocity_wh = momentum * velocity_wh + eta * x * delta_h';
    
    wo = wo + velocity_wo;
    wh = wh + velocity_wh;
end

fig5 = figure('Position', [100, 100, 600, 400], 'Color', 'white');
plot(ee, 'b-', 'LineWidth', 1.5);
xlabel('Epoch', 'FontSize', 12);
ylabel('Loss', 'FontSize', 12);
title('ReLU Network with Momentum', 'FontSize', 14);
grid on;
set(gca, 'Color', 'white', 'GridColor', [0.8 0.8 0.8]);
saveas(fig5, fullfile('ProjectFigures', 'relu_one _layer _momentum . png'));
saveas(fig5, fullfile('ProjectFigures', 'relu_one _layer _momentum . fig'));
fprintf('  Saved: ProjectFigures/relu_one_layer _momentum . png\n');
close(fig5);

%% l=2 ReLU + He Initialization
fprintf('\n Running two-layer ReLU with He initialization...\n');
clear; close all;
load MNIST_train

x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

input_size = 784;
hidden1_size = 300;
hidden2_size = 150;
output_size = 10;

wh1 = randn(input_size, hidden1_size) * sqrt(2/input_size);
wh2 = randn(hidden1_size, hidden2_size) * sqrt(2/hidden1_size);
wo = randn(hidden2_size, output_size) * sqrt(2/hidden2_size);

int = 1000;
eta = 0.1;
momentum = 0.9;

velocity_wo = zeros(size(wo));
velocity_wh2 = zeros(size(wh2));
velocity_wh1 = zeros(size(wh1));
ee = zeros(1, int);

for k = 1:int
    vh1 = wh1' * x;
    yh1 = max(0, vh1);
    vh2 = wh2' * yh1;
    yh2 = max(0, vh2);
    vo = wo' * yh2;
    vo = vo - max(vo, [], 1);
    evo = exp(vo);
    yo = evo ./ (ones(output_size,1) * sum(evo, 1));
    
    loss = -mean(sum(d .* log(yo + 1e-12), 1));
    ee(k) = loss;
    
    delta_o = (d - yo) / 60000;
    delta_h2 = (wo * delta_o) .* (vh2 > 0);
    delta_h1 = (wh2 * delta_h2) .* (vh1 > 0);
    
    velocity_wo = momentum * velocity_wo + eta * yh2 * delta_o';
    velocity_wh2 = momentum * velocity_wh2 + eta * yh1 * delta_h2';
    velocity_wh1 = momentum * velocity_wh1 + eta * x * delta_h1';
    
    wo = wo + velocity_wo;
    wh2 = wh2 + velocity_wh2;
    wh1 = wh1 + velocity_wh1;
end

fig6 = figure('Position', [100, 100, 600, 400], 'Color', 'white');
plot(ee, 'b-', 'LineWidth', 1.5);
xlabel('Epoch', 'FontSize', 12);
ylabel('Loss', 'FontSize', 12);
title('Two-Layer ReLU with He Initialization', 'FontSize', 14);
grid on;
set(gca, 'Color', 'white', 'GridColor', [0.8 0.8 0.8]);
saveas(fig6, fullfile('ProjectFigures', 'relu_two _layer _momentum . png'));
saveas(fig6, fullfile('ProjectFigures', 'relu_two _layer _momentum . fig'));
fprintf('  Saved: ProjectFigures/relu_two_layer _momentum . png\n');
close(fig6);


% Compression Figs

fprintf('\n--- PART 3: Initialization Comparison Figures ---\n');

%% 3.1 Initialization Comparison
fprintf('Running initialization comparison...\n');
clear; close all;
load MNIST_train

x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

x_small = x(:, 1:10000);
d_small = d(:, 1:10000);
labels_small = labelstrain(1:10000);

hidden_size = 300;
output_size = 10;
int = 200;
eta = 0.1;
momentum = 0.9;

init_methods = {'Uniform [-0.5,0.5]', 'Gaussian N (0,0.1)', 'Xavier', 'He'};
colors = {'r-', 'g-', 'b-', 'k-'};
loss_curves = zeros(4, int);
accuracies = zeros(1, 4);

for method = 1:4
    switch method
        case 1
            wo = rand(hidden_size, output_size) - 0.5;
            wh = rand(28*28, hidden_size) - 0.5;
        case 2
            wo = 0.1 * randn(hidden_size, output_size);
            wh = 0.1 * randn(28*28, hidden_size);
        case 3
            wo = randn (hidden_size, output_size) / sqrt(hidden_size);
            wh = randn (28*28, hidden_size) / sqrt(28*28);
        case 4
            wo = randn(hidden_size, output_size) * sqrt(2/hidden_size);
            wh = randn(28*28, hidden_size) * sqrt(2/28*28);
    end
    
    velocity_wo = zeros(size(wo));
    velocity_wh = zeros(size(wh));
    
    for k = 1:int
        vh = wh' * x_small;
        yh = max(0, vh);
        vo = wo' * yh;
        vo = vo - max(vo, [], 1);
        evo = exp(vo);
        yo = evo ./ (ones(output_size,1) * sum(evo, 1));
        
        loss = -mean(sum(d_small .* log(yo + 1e-12), 1));
        loss_curves(method, k) = loss;
        
        delta_o = (d_small - yo) / 10000;
        delta_h = (wo * delta_o) .* (vh > 0);
        
        velocity_wo = momentum * velocity_wo + eta * yh * delta_o';
        velocity_wh = momentum * velocity_wh + eta * x_small * delta_h';
        
        wo = wo + velocity_wo;
        wh = wh + velocity_wh;
    end
    
    yh_test = max(0, wh' * x_small);
    vo_test = wo' * yh_test;
    vo_test = vo_test - max(vo_test, [], 1);
    evo_test = exp(vo_test);
    yo_test = evo_test ./ (ones(output_size,1) * sum(evo_test, 1));
    [~, pred] = max(yo_test);
    accuracies(method) = sum (pred == (labels_small + 1)') / 10000;
end

% Create comp figure
fig7 = figure('Position', [100, 100, 1000, 800], 'Color', 'white');

subplot(2,2,1);
hold on;
for method = 1:4
    plot(1:int, loss_curves(method, :), colors{method}, 'LineWidth', 1.5);
end
hold off;
xlabel('Epoch'); ylabel('Loss');
title('Loss Convergence');
legend(init_methods, 'Location', 'northeast');
grid on;

subplot(2,2,2);
bar(accuracies * 100);
set(gca, 'XTickLabel', init_methods);
xtickangle(45);
ylabel('Test Accuracy (%)');
title('Accuracy after 200 Epochs');
grid on;

subplot(2,2,3);
hold on;
for method = 1:4
    semilogy(1:int, loss_curves(method, :), colors{method}, 'LineWidth', 1.5);
end
hold off;
xlabel('Epoch'); ylabel('Log Loss');
title('Loss (log scale)');
legend(init_methods, 'Location', 'northeast');
grid on;

subplot(2,2,4);
final_losses = loss_curves(:, end);
bar(final_losses);
set(gca, 'XTickLabel', init_methods);
xtickangle(45);
ylabel('Final Loss');
title('Final Loss Comparison');
grid on;

sgtitle('Initialization Method Comparison');
saveas(fig7, fullfile('ProjectFigures', 'initialization_comparison _full . png'));
saveas(fig7, fullfile('ProjectFigures', 'initialization_comparison _full . fig'));
fprintf('  Saved: ProjectFigures/initialization_comparison_full . png\n');
close(fig7);

% Indv curves figure
fig8 = figure('Position', [100, 100, 1200, 800], 'Color', 'white');
for method = 1:4
    subplot(2,2,method);
    plot(1:int, loss_curves(method, :), colors{method}, 'LineWidth', 2);
    xlabel('Epoch'); ylabel('Loss');
    title(init_methods{method});
    grid on;
    text(int*0.6, loss_curves(method, int)*1.5, ...
         sprintf('Acc: % .2f %%', accuracies(method)*100), ...
         'FontSize', 11, 'FontWeight', 'bold');
end
sgtitle('Individual Training Curves by Initialization Method');
saveas(fig8, fullfile('ProjectFigures', 'initialization_individual _curves . png'));
saveas(fig8, fullfile('ProjectFigures', 'initialization_individual _curves . fig'));
fprintf('  Saved: ProjectFigures/initialization_individual_curves . png\n');
close(fig8);

fprintf('\n========================================\n');
fprintf('ALL FIGURES GENERATED SUCCESSFULLY!\n');
fprintf('Figures saved in: ProjectFigures folder \n');
fprintf('========================================\n');
