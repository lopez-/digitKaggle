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

[cost grad] = nnCostFunction(initial_nnParams,n,hidden_layer_size,num_labels, X, y, 0.1);