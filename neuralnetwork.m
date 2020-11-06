clc
clear
%%
data = xlsread('train.xlsx');
data2 = data(1:27,1:3);
input = [data2]';
%%
target = data(1:27,4:5);
target = [target]';

datatest=data(1:12 , 6:8);
output = [datatest]';
target1 = data(1:12 , 9:10);
target1 = [target1]';

%%
%parameter
nilai_lr=0.25;
nilai_epoch=100;
nilai_eror=0.001;
nilai_hide=15;


%Building the Neural Network Classifier
net = newff(input,target, nilai_hide, {'logsig'});  

%training teknik heuristik hal.150 
net.trainParam.epochs=nilai_epoch;  %set maksimum epoch
net.trainParam.goal=nilai_eror;  %set tolersi error
net.trainParam.lr=nilai_lr;

[net,tr]= train(net,input,target); %batch mode backprop

%output --> hasil simulasi dari data yang dilatih
[out1]  = sim(net,input);

hasil=round(out1);  %hasil dibulatkan dengan bilangan terdekat
target;


%%
%testing
inputtest=datatest';
[out2]  = sim(net,inputtest);
out2=round(out2); %hasil dibulatkan dengan bilangan terdekat

%% hasil data training
training=[hasil' target']
%perf = perform(net,hasil,target)


%hasil data testing
testing = [out2' target1']
%perf = perform(net,out2,target1)
