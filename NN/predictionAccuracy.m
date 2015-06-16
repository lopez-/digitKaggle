function a = predictionAccuracy(prediction, actuals)

fprintf('\nTraining Set Accuracy: %f\n', mean(double(prediction == actuals)) * 100);

end;