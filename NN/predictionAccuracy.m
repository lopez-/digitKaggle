function [accu] = predictionAccuracy(prediction, actuals)

accu = (mean(double(prediction == actuals)) * 100);

end;