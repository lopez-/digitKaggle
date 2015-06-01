% Load training data
training = csvread('train.csv',1,0);

% Define main variables
y = training(:,1);
% Convert all zero's to ten's, to make it more compatible with Octave indexing (starting from 1)
y( y == 0 ) = 10; 

X = training(:,2:end);

max_iterations = 200
iterations = 10

while iterations < max_iterations
  
  counter = iterations/10
  
  % Run oneVsAll to get an optimized vector of theta's. This will be a matrix k x m (10x784)
  opt_theta = oneVsAll(X, y, 10, 0.1, iterations);

  % With an optimized theta (opt_theta), run predictOneVsAll to get a vector of predictions based 
  % the features of X. This will be a vector m x 1 
  trainingPrediction = predictOnevsAll(opt_theta, X);

  my_prediction = mean(double(trainingPrediction == y)) * 100;
  
  prediction_log(counter) = my_prediction;
  iteration_log(counter) = iterations;
  
  iterations = iterations + 10;
  
  if iterations == max_iterations
    plot(prediction_log, iteration_log)
  end
  
end
