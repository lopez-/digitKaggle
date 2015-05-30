% Load training data
training = csvread('train.csv',1,0);

% Define main variables
y = training(:,1);
% Convert all zero's to ten's, to make it more compatible with Octave indexing (starting from 1)
y( y == 0 ) = 10; 

X = training(:,2:end);

% Run oneVsAll to get an optimized vector of theta's. This will be a matrix k x m (10x784)
opt_theta = oneVsAll(X, y, 10, 0.1, 100);

% With an optimized theta (opt_theta), run predictOneVsAll to get a vector of predictions based 
% the features of X. This will be a vector m x 1 
trainingPrediction = predictOnevsAll(opt_theta, X);

% Print accuracy of prediction (against y)
accuracy(trainingPrediction, y);

% Load test data
test = csvread('test.csv', 1, 0);

% Convert 0's to 10's
test(test == 0) = 10;

% Predict data on the test dataset
testPrediction = predictOnevsAll(opt_theta, test);

% Convert back from 10's to 0's
testPrediction(testPrediction==10) = 0;

nrowTestPrediction = size(testPrediction ,1);

toSubmit = [[1:nrowTestPrediction]' testPrediction];

 headers = {'ImageId','Label'};
 
 csvwrite_with_headers('myThirdPrediction.csv',toSubmit,headers);
