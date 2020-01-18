clear all;clc,close all
%% Extrame Learning Machine Algorithm
% Nonlinear Regression
% Raþit EVDÜZEN 24-Aug-2017 

%% Regression
load data.mat

X = X(1:200,:);
Y = Y(1:200,:);
%% Normalization Case
X = [X-min(X)]./[max(X)-min(X)];
Y = [Y-min(Y)]/[max(Y)-min(Y)];

%% Training
for i=1:length(Y)
    [Elm_output] = elm(X,Y,i);
    rmse(i) = sqrt(mse(Y'-Elm_output));
end

%% PLOT DATA
subplot(121)
plot(Y,'r','LineWidth',2)
hold on,grid minor,xlabel('Number Of Data'),ylabel('')
plot(Elm_output,'k--','LineWidth',2),title('Extrame Learning Machine Nonlinear Regression')
legend('Training Data','ELM Output')

subplot(122)
plot(rmse,'LineWidth',2);
xlabel('Number Of Neurons'),ylabel('RMSE')
grid minor,title(['root mean square error ',num2str(rmse(1,end))])