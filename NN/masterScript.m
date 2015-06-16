% Load training data
training = csvread('train.csv',1,0);

% Define main variables
y = training(:,1);

X = training(:,2:end);

m = size(X)(1);
n = size(X)(2);
hidden_layer_size = 25;
num_labels = 10;

initial_Theta1 = randInitializeWeights(n,hidden_layer_size);

initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels);

initial_nnParams = [initial_Theta1(:); initial_Theta2(:)];

options = optimset('MaxIter', 200);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, n, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nnParams, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (n + 1)), ...
                 hidden_layer_size, (n + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (n + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

p = nnPredict(nn_params, X, hidden_layer_size, num_labels);

p ( p == 10 ) = 0;

predictionAccuracy(p,y);

test = csvread('test.csv',1,0);

pTest = nnPredict(nn_params, test, 25, 10);

pTest ( pTest == 10 ) = 0;

rowsPrediction = size(pTest)(1);

toSubmit = [[1:rowsPrediction]' pTest];

headers = {'ImageId', 'Label'};

csvwrite_with_headers("firstPredictionNN",toSubmit, headers);