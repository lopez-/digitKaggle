function a = accuracy(prediction, y)

 fprintf('\nTraining Set Accuracy: %f\n', mean(double(prediction == y)) * 100);

;