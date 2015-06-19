clear ; close all; clc

% Load training data
training = csvread('train.csv',1,0);

% Define main variables
y = training(:,1);
X = training(:,2:end);

% Define structure of neural network
m = size(X)(1);
n = size(X)(2);
hidden_layers = 4;
input_layer_size = n;
hidden_layer_size = 25;
num_labels = 10;

% Random initialize Weights of NN
Thetas = {};
Thetas(1) = randInitializeWeights(n,hidden_layer_size);

if (hidden_layers==1)
  Thetas(2) = randInitializeWeights(hidden_layer_size, num_labels);
else
  for i=1:(hidden_layers-1)
    Thetas(i+1) = randInitializeWeights(hidden_layer_size, hidden_layer_size);
  end
  Thetas(hidden_layers+1) = randInitializeWeights(hidden_layer_size, num_labels);
end

initial_nnparams = [];
for i=1:(hidden_layers+1)
  initial_nnparams = [initial_nnparams; Thetas{i}(:)];
end

fprintf('\n Press enter to ensure gradients are correctly implemented without regularization.\n');
pause;

% Check gradient is correct not considering regularization
checkNNGradients;

fprintf('\n Press enter to ensure gradients are correctly implemented with regularization.\n');
pause;

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% If implementation is correct, next step is to train the NN
fprintf('\n Press enter to train the NN.\n');
pause;

options = optimset('MaxIter', 200);

% Set lambda value
lambda = 0.5;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, hidden_layers, n, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nnparams, options);

% Reshape nn_params into a cell of Thetas
trainedThetas = {};

trainedThetas(1) = reshape(nn_params(1:((input_layer_size+1)*hidden_layer_size)), hidden_layer_size, (input_layer_size+1));
start_to_reshape = (input_layer_size+1)*hidden_layer_size;

if (hidden_layers==1)
  trainedThetas(2) = reshape(nn_params((start_to_reshape+1):end), num_labels, (hidden_layer_size+1));
else
  for i=1:(hidden_layers-1)
    trainedThetas(i+1) = reshape(nn_params((start_to_reshape+1):(start_to_reshape+(hidden_layer_size*(hidden_layer_size+1)))), hidden_layer_size, (hidden_layer_size+1));
    start_to_reshape = (start_to_reshape+(hidden_layer_size*(hidden_layer_size+1)));
  end
  trainedThetas(hidden_layers+1) = reshape(nn_params((start_to_reshape+1):end), num_labels,(hidden_layer_size+1));
end

fprintf('\n Press enter to make a prediciton on the training dataset.\n');
pause;

% Use the trained classifiers to predict the values on the training dataset
p = nnPredict(nn_params, X, hidden_layers, input_layer_size, hidden_layer_size, num_labels);

% Convert back digits identified as 10 to 0
p ( p == 10 ) = 0;

% Calculate the accuracy of the prediction given the actuals, which is the vector y
pTraining = predictionAccuracy(p,y);
fprintf('\nTraining Set Accuracy: %f\n', pTraining);

fprintf('\n Press enter to read the test dataset and make a prediction on it.\n');
pause;

% Read the test dataset
test = csvread('test.csv',1,0);

% Use the trained classifiers to make a prediciton based on the test dataset
pTest = nnPredict(nn_params, test, hidden_layers, input_layer_size, hidden_layer_size, num_labels);

% Convert back digits identified as 10 to 0
pTest ( pTest == 10 ) = 0;

% Store the prediciton on a CSV
rowsPrediction = size(pTest)(1);
toSubmit = [[1:rowsPrediction]' pTest];
headers = {'ImageId', 'Label'};
csvwrite_with_headers("3PredictionNN",toSubmit, headers);