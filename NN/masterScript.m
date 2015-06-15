% Load training data
training = csvread('train.csv',1,0);

% Define main variables
y = training(:,1);

X = training(:,2:end);