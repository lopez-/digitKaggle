function [p] = nnPredict(nnParams, X, hidden_layer_size, num_labels)

m = size(X)(1);
n = size(X)(2);

sizeTheta1 = [hidden_layer_size (n+1)];
sizeTheta2 = [num_labels (hidden_layer_size+1)];

Theta1 = reshape(nnParams(1:(sizeTheta1(1)*sizeTheta1(2))),sizeTheta1(1), sizeTheta1(2));
Theta2 = reshape(nnParams(((sizeTheta1(1)*sizeTheta1(2))+1):end),sizeTheta2(1), sizeTheta2(2));

A1 = X;

A1_exp = [ones(m,1) X];

A2 = sigmoid(A1_exp*Theta1');

A2_exp = [ones(m,1) A2];

A3 = sigmoid(A2_exp*Theta2');

[maxs p] = max(A3, [], 2);

end;
