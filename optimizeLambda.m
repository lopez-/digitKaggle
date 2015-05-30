% Load training data
training = csvread('train.csv',1,0);

% Define main variables
y = training(:,1);
% Convert all zero's to ten's, to make it more compatible with Octave indexing (starting from 1)
y( y == 0 ) = 10; 

X = training(:,2:end);

lambda = 0.05
counter = 1

while lambda < 0.25
  
  % Run oneVsAll to get an optimized vector of theta's. This will be a matrix k x m (10x784)
  opt_theta = oneVsAll(X, y, 10, lambda, 180);

  % With an optimized theta (opt_theta), run predictOneVsAll to get a vector of predictions based 
  % the features of X. This will be a vector m x 1 
  trainingPrediction = predictOnevsAll(opt_theta, X);

  my_prediction = mean(double(trainingPrediction == y)) * 100;
  
  prediction_log(counter) = my_prediction;
  lambda_log(counter) = lambda;
  counter = counter + 1;
  
  lambda = lambda + 0.05;
  
  if lambda == 0.20
    plot(lambda_log, prediction_log);
  end
  
end