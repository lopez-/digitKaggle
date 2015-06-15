% Load training data
training = csvread('train.csv',1,0);

% Define main variables
y = training(:,1);

X = training(:,2:end);

m = size(X)(1);
n = size(X)(2);
hidden_layer_size = 25;
num_labels = 10;

Theta1 = rand(25,785);
Theta2 = rand(10,26);

nnParams = [Theta1(:); Theta2(:)];

[cost grad] = nnCostFunction(nnParams,n,hidden_layer_size,num_labels, X, y, 0.1);