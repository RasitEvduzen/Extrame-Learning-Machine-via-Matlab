clear all;clc,close all
%% Extrame Learning Machine Algorithm
% Multiple Input Single Output Nonlinear Regression 
% Raþit EVDÜZEN 1-Jan-2017
% Edit : 19-Jan-2020

%% Regression
load data.mat

X = [X - min(X)] ./[max(X) - min(X)];
Y = [Y - min(Y)] ./[max(Y) - min(Y)];

TrainingInput  = X(1:2:end,:);
TrainingOutput = Y(1:2:end,:);
TestingInput  = X(2:2:end,:);
TestingOutput = Y(2:2:end,:);

TestPerformance = inf;
for i = 1:length(TrainingOutput)
    NuberOfNeuron = i;  % Number of Neuron
    
    %% Training Case
    Witr = 2*rand(NuberOfNeuron,size(TrainingInput,2))-1;  % generate a random input weights
    % H = radbas(input_weights*X');   % Radial Basis Function using Activation Function
    % H = sin(input_weights*X');      % Sin basis Function
    % H = cos(input_weights*X');      % Cos basis Function
    Htr = tanh(Witr*TrainingInput');       % TanH
    % H = 1 ./ (1 + exp(-input_weights));  % Sigmoid Function
    
    %  calculate the output weights
    Wo = pinv(Htr') * TrainingOutput ; % (LSE solve) pseudoinverse of matrix
    
    ElmOutputTr = (Htr' * Wo)' ; % calculate the actual output
    rmsetr(i) = sqrt(mse(TrainingOutput'-ElmOutputTr));
    
    %% Testing Case
    Witr = 2*rand(NuberOfNeuron,size(TestingInput,2))-1;   % generate a random input weights
    Hts = tanh(Witr*TestingInput');                        % TanH
    ElmOutputTs= (Hts' * Wo)' ; % calculate the actual output
    ElmOutputTs = (ElmOutputTs - min(ElmOutputTs)) / (max(ElmOutputTs) - min(ElmOutputTs));
    rmsets(i) = sqrt(mse(TestingOutput'-ElmOutputTs));
    
    if rmsets(i) < TestPerformance
        TestPerformance = rmsets(i);
        BestWo = Wo;
        BestElmOutputTs = ElmOutputTs;
    end
    
end

%% PLOT DATA
subplot(221)
plot(TrainingOutput,'r','LineWidth',2)
hold on,grid minor,xlabel('Number Of Data'),ylabel('')
plot(ElmOutputTr,'k--','LineWidth',2),title('Training RMSE')
legend('Training Data','ELM Output')

subplot(222)
plot(TestingOutput,'r','LineWidth',2)
hold on,grid minor,xlabel('Number Of Data'),ylabel('')
plot(BestElmOutputTs,'k--','LineWidth',2),title('Best Testing Model ')
legend('Testing Data','ELM Output')

subplot(223)
plot(rmsetr,'LineWidth',2), grid minor,xlabel('Number Of Neuron'),ylabel('Rmse Training')
title(['Min Training RMSE',num2str(min(rmsetr))])

subplot(224)
plot(rmsets,'LineWidth',2), grid minor,xlabel('Number Of Neuron'),ylabel('Rmse Testing')
title(['Min Testing RMSE',num2str(min(rmsets))]),hold on
scatter(size(BestWo,1),TestPerformance,'r','filled')