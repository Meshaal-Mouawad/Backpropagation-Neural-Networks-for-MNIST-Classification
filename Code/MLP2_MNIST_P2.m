%MLP - one hidden layer (ReLU), output layer (Softmax + Cross Entropy)
clear; close all
load MNIST_train

x = reshape(imagestrain,[28*28 60000])/255;
d = zeros(10, 60000);
classes = 0:9;
d = (labelstrain == classes); d = double(d');

wo = randn(300,10)*sqrt(1/784);
wh = randn(28*28,300)*sqrt(1/784); 

int = 1000;
eta = 0.5;

%batch learning
for k = 1:int
    vh = wh'*x;
    yh = max(0, vh);

    vo = wo'*yh;
    vo = vo - max(vo,[],1);
    evo = exp(vo);
    yo = evo./(ones(10,1)*sum(evo,1));
    e = - mean(sum(d.*log(yo + 1e-12),1));
    delta_o = (d-yo)/60000;
    dwo = eta*yh*delta_o';

    delta_h = (wo*delta_o).*double(vh>0);
    dwh = eta*x*delta_h';

    wo = wo + dwo;
    wh = wh + dwh;
    
    ee(k) = e;
end
figure; plot(ee(2:end))

vo_final = wo'*max(0,wh'*x); 
vo_final = vo_final - max(vo_final,[],1);
evo_final = exp(vo_final);
yo_final = evo_final./(ones(10,1)*sum(evo_final,1));
[m1 m2] = max(yo_final);
accuracy_train = sum(m2==(labelstrain+1)')/60000

%test
load MNIST_test
xtest = reshape(images,[28*28 10000])/255;
vo_final_test = wo'*max(0,wh'*xtest); vo_final_test = vo_final_test - max(vo_final_test,[],1);
evo_final_test = exp(vo_final_test);
yo_final_test = evo_final_test./(ones(10,1)*sum(evo_final_test,1));
[m1 m2] = max(yo_final_test);
accuracy_test = sum(m2==(labels+1)')/10000