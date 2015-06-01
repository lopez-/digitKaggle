function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


J = (1/m)*((-y'*log(sigmoid(X*theta)))-((1-y)'*(log(1-sigmoid(X*theta))))) + (lambda/(2*m))*(sum(theta(2:end).^2));

grad = (1/m)*((sigmoid(X*theta)-y)'*X)' + (lambda/m)*theta;

grad(1) = (1/m)*((sigmoid(X*theta)-y)'*X)'(1);

grad = grad(:);

end