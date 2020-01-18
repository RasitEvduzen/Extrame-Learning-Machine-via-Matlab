function [output] = elm(X,Y,neuron)

input_weights = rand(neuron,size(X,2))*2-1;  % generate a random input weights

% calculate the hidden layer, using Activation Function
% H = radbas(input_weights*X');        % Radial Basis Function 
% H = sin(input_weights*X');           % Sin basis Function
% H = cos(input_weights*X');           % Cos basis Function
H = tanh(input_weights*X');            % TanH
% H = 1 ./ (1 + exp(-input_weights));  % Sigmoid Function

%  calculate the output weights beta
B = pinv(H') * Y ; % (LSE solve) pseudoinverse of matrix

output = (H' * B)' ; % calculate the actual output
end