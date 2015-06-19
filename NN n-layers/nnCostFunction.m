function[J grad] = nnCostFunction(nn_params, hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

y ( y == 0 ) = 10;

y_matrix = eye(num_labels)(y,:);

m = size(X)(1);
n = size(X)(2);


% Reshape nn_params into a cell of Thetas
Thetas = {};

Thetas(1) = reshape(nn_params(1:((input_layer_size+1)*hidden_layer_size)), hidden_layer_size, (input_layer_size+1));
start_to_reshape = (input_layer_size+1)*hidden_layer_size;

if (hidden_layers==1)
  Thetas(2) = reshape(nn_params((start_to_reshape+1):end), num_labels, (hidden_layer_size+1));
else
  for i=1:(hidden_layers-1)
    Thetas(i+1) = reshape(nn_params((start_to_reshape+1):(start_to_reshape+(hidden_layer_size*(hidden_layer_size+1)))), hidden_layer_size, (hidden_layer_size+1));
    start_to_reshape = (start_to_reshape+(hidden_layer_size*(hidden_layer_size+1)));
  end
  Thetas(hidden_layers+1) = reshape(nn_params((start_to_reshape+1):end), num_labels,(hidden_layer_size+1));
end


% Get A's and A_exp's
A = {};
A_exp = {};
for i=1:(length(Thetas)+1)
  if (i==1)
    A(i) = X;
    A_exp(i) = [ones(m,1) A{i}];
  elseif (i==(length(Thetas)+1))
    A(i) = sigmoid(A_exp{i-1}*Thetas{i-1}');
  else
    A(i) = sigmoid(A_exp{i-1}*Thetas{i-1}');
    A_exp(i) = [ones(m,1) A{i}];
  end
end

tempCost = 0;

for i = 1:num_labels
  
  tempY = y_matrix(:,i);
  
  tempA = A{length(A)}(:,i);
  
  classCost = (-tempY'*log(tempA)) - ((1-tempY)'*log(1-tempA));
  
  tempCost = tempCost + classCost;
  
end;

regularization = 0;
for i=1:(length(Thetas))
  regularization = regularization + sum(Thetas{i}(:,2:end)(:).^2);
end

J = ((1/m)*tempCost) + ((lambda/(2*m))*regularization);

d = {};
for i=(length(Thetas)+1):-1:2
  if (i==(length(Thetas)+1))
    d(i) = A{i}-y_matrix;
  else
    d(i) = (d{i+1}*Thetas{i}(:,2:end)).*sigmoidGradient(A_exp{i-1}*Thetas{i-1}');
  end
end

D = {};
for i=1:(length(Thetas))
  D(i) = d{i+1}'*A_exp{i};
end

tempTheta = {};
for i=1:(length(Thetas))
  tempTheta(i) = (lambda/m)*Thetas{i};
  tempTheta{i}(:,1) = 0;
end

Theta_grad = {};
for i=1:(length(Thetas))
  Theta_grad(i) = ((1/m)*D{i})+tempTheta{i};
end

grad = [];
for i=1:(length(Theta_grad))
  grad = [grad; Theta_grad{i}(:)];
end

end;