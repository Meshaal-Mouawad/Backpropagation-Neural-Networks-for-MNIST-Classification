%MLP - one hidden layer
clear; close all
load MNIST_train

x = reshape(imagestrain,[28*28 60000])/255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes); d = double(d');

wo = rand(300,10)-0.5;
wh = rand(28*28,300)-0.5; 

sigma = 0.01;
int = 1000;
eta = 0.001;

%batch learning
for k = 1:int
    vh = wh'*x;
    yh = 1./(1+exp(-sigma*vh));

    vo = wo'*yh;
    yo = 1./(1+exp(-sigma*vo));

    e = d - yo;

    delta_o = e.*yo.*(1-yo);
    dwo = eta*yh*delta_o';

    delta_h = (wo*delta_o).*yh.*(1-yh);
    dwh = eta*x*delta_h';

    wo = wo + dwo;
    wh = wh + dwh;

    ee(k) = norm(e);
end
figure; plot(ee(2:end));
yo_final = 1./(1+exp(-sigma*(wo'*(1./(1+exp(-sigma*(wh'*x)))))));
[m1 m2] = max(yo_final);
accuracy_train = sum(m2==(labelstrain+1)')/60000

%test
load MNIST_test
xtest = reshape(images,[28*28 10000])/255;
yo_final_test = 1./(1+exp(-sigma*(wo'*(1./(1+exp(-sigma*(wh'*xtest)))))));
[m1 m2] = max(yo_final_test);
accuracy_test = sum(m2==(labels+1)')/10000