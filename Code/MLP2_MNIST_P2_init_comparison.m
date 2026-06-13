(* ::Package:: *)

% MLP2_MNIST _P2 _init _comparison . m
% Compare different initialization methods for ReLU network
% Uniform, Gaussian/Xavier, and He initialization

clear; close all
load MNIST_train

% subset comparison
fprintf('Preparing data...\n');
x = reshape (imagestrain, [28*28 60000]) / 255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes);
d = double(d');

% Use 10000 samples
x_small = x(:, 1:10000);
d_small = d(:, 1:10000);
labels_small = labelstrain(1:10000);

% Net vars
hidden_size = 300;
output_size = 10;
int = 200;  % Fewer epochs for comparison
eta = 0.1;
momentum = 0.9;

% the results
init_methods = {'Uniform [-0.5,0.5]', 'Gaussian N (0,0.1)', 'Xavier', 'He'};
accuracies = zeros(1, 4);
loss_curves = zeros(4, int);
grad_norms = zeros(4, int);

fprintf('\n=== INITIALIZATION COMPARISON ===\n');
fprintf('Testing 4 different initialization methods...\n \n');

for method = 1:4
    fprintf('Testing % s...\n', init_methods{method});
    
    % methods 
    switch method
        case 1  % Uniform
            wo = rand(hidden_size, output_size) - 0.5;
            wh = rand(28*28, hidden_size) - 0.5;
            fprintf('  Weight range: [% .2f, % .2f]\n', min(wo(:)), max(wo(:)));
            
        case 2  % Gaussian N(0, 0.1)
            wo = 0.1 * randn(hidden_size, output_size);
            wh = 0.1 * randn(28*28, hidden_size);
            fprintf('  Weight std: % .2f \n', std(wo(:)));
            
        case 3  % Xavier 
            wo = randn (hidden_size, output_size) / sqrt(hidden_size);
            wh = randn (28*28, hidden_size) / sqrt(28*28);
            fprintf('  Xavier scale: 1/sqrt(% d)\n', hidden_size);
            
        case 4  % He
            wo = randn(hidden_size, output_size) * sqrt(2/hidden_size);
            wh = randn(28*28, hidden_size) * sqrt(2/28*28);
            fprintf('  He scale: sqrt(2/%d)\n', hidden_size);
    end
    
    % Momntm Velocities
    velocity_wo = zeros(size(wo));
    velocity_wh = zeros(size(wh));
    
    % Training
    for k = 1:int
        % Fwd propagation
        vh = wh' * x_small;
        yh = max(0, vh);
        
        vo = wo' * yh;
        vo = vo - max(vo, [], 1);
        evo = exp(vo);
        yo = evo ./ (ones(output_size,1) * sum(evo, 1));
        
        % Loss
        loss = -mean(sum(d_small .* log(yo + 1e-12), 1));
        loss_curves(method, k) = loss;
        
        % Back-propagation
        delta_o = (d_small - yo) / 10000;
        delta_h = (wo * delta_o) .* (vh > 0);
        
        % Gradient nor 
        grad_norms(method, k) = norm(delta_o);
        
        % Momentum updates
        velocity_wo = momentum * velocity_wo + eta * yh * delta_o';
        velocity_wh = momentum * velocity_wh + eta * x_small * delta_h';
        
        wo = wo + velocity_wo;
        wh = wh + velocity_wh;
    end
    
    % Accur
    yh_test = max(0, wh' * x_small);
    vo_test = wo' * yh_test;
    vo_test = vo_test - max(vo_test, [], 1);
    evo_test = exp(vo_test);
    yo_test = evo_test ./ (ones(output_size,1) * sum(evo_test, 1));
    [~, pred] = max(yo_test);
    accuracies(method) = sum (pred == (labels_small + 1)') / 10000;
    
    fprintf('  Accuracy after % d epochs: % .2f %% \n', int, accuracies(method)*100);
end

% results Figure
figure('Position', [100, 100, 1000, 800]);

% Plot 1: Loss convergence
subplot(2,2,1);
colors = {'r-', 'g-', 'b-', 'k-'};
hold on;
for method = 1:4
    plot(1:int, loss_curves(method, :), colors{method}, 'LineWidth', 1.5);
end
hold off;
xlabel('Epoch');
ylabel('Cross-Entropy Loss');
title('Loss Convergence by Initialization Method');
legend(init_methods, 'Location', 'northeast');
grid on;

% Plot 2: Final accr
subplot(2,2,2);
bar(accuracies * 100);
set(gca, 'XTickLabel', init_methods);
xtickangle(45);
ylabel('Test Accuracy (%)');
title('Accuracy after 200 Epochs');
grid on;

% Plot 3: Gradient norms
subplot(2,2,3);
hold on;
for method = 1:4
    plot(1:int, grad_norms(method, :), colors{method}, 'LineWidth', 1);
end
hold off;
xlabel('Epoch');
ylabel('Gradient Norm');
title('Gradient Magnitudes');
legend(init_methods, 'Location', 'northeast');
grid on;

% Plot 4: Loss at epoch = 200
subplot(2,2,4);
final_losses = loss_curves(:, end);
bar(final_losses);
set(gca, 'XTickLabel', init_methods);
xtickangle(45);
ylabel('Final Loss (Epoch 200)');
title('Final Loss Comparison');
grid on;

sgtitle('Impact of Weight Initialization on ReLU Network Performance');
saveas(gcf, 'initialization_comparison _full . png');

% Prnt table
fprintf('\n=== INITIALIZATION COMPARISON SUMMARY ===\n');
fprintf('%-20s | %-15s | %-15s | %-15s \n', 'Method', 'Final Loss', 'Gradient Norm', 'Accuracy');
fprintf('% s \n', repmat('-', 1, 70));
for method = 1:4
    fprintf('%-20s | %-15.4f | %-15.2e | %-15.2f %% \n', ...
            init_methods{method}, ...
            loss_curves(method, end), ...
            grad_norms(method, end), ...
            accuracies(method)*100);
end

% additional: Training curves / method
figure('Position', [100, 100, 1200, 800]);
for method = 1:4
    subplot(2,2,method);
    plot(1:int, loss_curves(method, :), colors{method}, 'LineWidth', 2);
    xlabel('Epoch');
    ylabel('Loss');
    title(init_methods{method});
    grid on;
    text(int*0.6, loss_curves(method, int)*1.5, ...
         sprintf('Acc: % .2f %%', accuracies(method)*100), ...
         'FontSize', 11, 'FontWeight', 'bold');
end
sgtitle('Individual Training Curves by Initialization Method');
saveas(gcf, 'initialization_individual _curves . png');

fprintf('\n=== DONE ===\n');
