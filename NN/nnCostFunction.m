function[J grad] = nnCostFunction(nnParams, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

y ( y == 0 ) = 10;

y_matrix = eye(num_labels)(y,:);

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

tempCost = 0;

for i = 1:num_labels
  
  tempY = y_matrix(:,i);
  
  tempA = A3(:,i);
  
  classCost = (-tempY'*log(tempA)) - ((1-tempY)'*log(1-tempA));
  
  tempCost = tempCost + classCost;
  
end;

regularization = sum(Theta1(:,2:end)(:).^2) + sum(Theta2(:,2:end)(:).^2);

J = ((1/m)*tempCost) + ((lambda/m)*regularization);

d3 = A3-y_matrix;

d2 = (d3*Theta2(:,2:end)).*sigmoidGradient(A1_exp*Theta1');

D1 = d2'*A1_exp;

D2 = d3'*A2_exp;

tempTheta1 = (lambda/m)*Theta1;

tempTheta2 = (lambda/m)*Theta2;

tempTheta1(:,1) = 0;

tempTheta2(:,1) = 0;

Theta1_grad = ((1/m)*D1)+tempTheta1;

Theta2_grad = ((1/m)*D2)+tempTheta2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end;