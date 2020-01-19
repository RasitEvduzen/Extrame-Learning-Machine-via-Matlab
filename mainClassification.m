clear all;clc,close all
%% Extrame Learning Machine Algorithm
% Multiple Input Single Output Classification Spiral Data Example
% Raþit EVDÜZEN 1-Jan-2017
% Edit: 19-Jan-2020

% Create Data
B = 4;
N = 200;
Tall = [];
for i=1:N/2
    theta = pi/2 + (i-1)*[(2*B-1)/N]*pi;
    Tall = [Tall , [theta*cos(theta);theta*sin(theta)]];
end
Tall = [Tall,-Tall];
Tmax = pi/2+[(N/2-1)*(2*B-1)/N]*pi;
X = [Tall]'/Tmax;
Y = [-ones(1,N/2), ones(1,N/2)]';

Performance = inf;
for i = 1:length(Y)
    NuberOfNeuron = i;  % Number of Neuron
    Wi = 2*rand(NuberOfNeuron,size(X,2))-1;  % generate a random input weights
    % H = radbas(input_weights*X');   % Radial Basis Function using Activation Function
    % H = sin(input_weights*X');      % Sin basis Function
    % H = cos(input_weights*X');      % Cos basis Function
    H = tanh(Wi*X');       % TanH
    % H = 1 ./ (1 + exp(-input_weights));  % Sigmoid Function
    %  calculate the output weights
    Wo = pinv(H') * Y ; % (LSE solve) pseudoinverse of matrix
    ElmOutput = (H' * Wo)' ; % calculate the actual output
    rmse(i) = sqrt(mse(Y'-ElmOutput));
    
end

%% PLOT DATA
subplot(121)
hold on,grid minor,xlabel('Number Of Data')
input = []; output = [];
for t1=-1:0.05:1
    for t2=-1:0.05:1
        input = [input; [t1,t2]];
        Ht = tanh(Wi*input');
        yhat = sign((Ht' * Wo)');
        output = [output yhat(1,end)];
    end
end
Iplus = find(output==+1);
Iminus = find(output==-1);
plot(input(Iplus,1),input(Iplus,2),'g.')
hold on
plot(input(Iminus,1),input(Iminus,2),'y.')

Iplus = find(Y==+1);
Iminus = find(Y==-1);
plot(X(Iplus,1), X(Iplus,2), 'bo')
plot(X(Iminus,1), X(Iminus,2), 'r*')
title('Nonlinear Classification')


subplot(122)
hold on,grid minor,xlabel('Number Of Neuron'),ylabel('Rmse')
plot(rmse,'b','LineWidth',2),title('RMSE Calculation')


