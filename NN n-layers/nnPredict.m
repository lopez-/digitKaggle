function [p] = nnPredict(nnParams, X, hidden_layers, input_layer_size, hidden_layer_size, num_labels)

m = size(X)(1);
n = size(X)(2);

% Reshape nnParams into a cell of Thetas
Thetas = {};

Thetas(1) = reshape(nnParams(1:((input_layer_size+1)*hidden_layer_size)), hidden_layer_size, (input_layer_size+1));
start_to_reshape = (input_layer_size+1)*hidden_layer_size;

if (hidden_layers==1)
  Thetas(2) = reshape(nnParams((start_to_reshape+1):end), num_labels, (hidden_layer_size+1));
else
  for i=1:(hidden_layers-1)
    Thetas(i+1) = reshape(nnParams((start_to_reshape+1):(start_to_reshape+(hidden_layer_size*(hidden_layer_size+1)))), hidden_layer_size, (hidden_layer_size+1));
    start_to_reshape = (start_to_reshape+(hidden_layer_size*(hidden_layer_size+1)));
  end
  Thetas(hidden_layers+1) = reshape(nnParams((start_to_reshape+1):end), num_labels,(hidden_layer_size+1));
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

[maxs p] = max(A{(length(Thetas)+1)}, [], 2);

end;
