function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X = [ones(m, 1) X];
%printf("Size of X: %f %f\n", size(X, 1), size(X, 2));
%printf("Size of Theta1: %f %f\n", size(Theta1, 1), size(Theta1, 2));
%printf("Size of Theta2: %f %f\n", size(Theta2, 1), size(Theta2, 2));
a1 = sigmoid(X*Theta1');
%printf("Size of a1: %f %f\n", size(a1, 1), size(a1, 2));

a1 = [ones(m, 1) a1];
%printf("Size of a1: %f %f\n", size(a1, 1), size(a1, 2));
h = sigmoid(a1*Theta2');
%printf("Size of h: %f %f\n", size(h, 1), size(h, 2));
y_matrix = full(sparse(1:numel(y), y, 1, numel(y), num_labels));
%printf("Size of y_matrix: %f %f\n", size(y_matrix, 1), size(y_matrix, 2));

theta1 = Theta1;
theta2 = Theta2;
theta1(:,[1]) = [];
theta2(:,[1]) = [];

%printf("Size of theta1: %f %f\n", size(theta1, 1), size(theta1, 2));
%printf("Size of theta2: %f %f\n", size(theta2, 1), size(theta2, 2));

regularization = (lambda/(2*m))*(sum(sum(theta1.^2)) + sum(sum(theta2.^2)));

%printf("start calculation J\n");
J = -(1/m)*sum(sum(y_matrix.*log(h) + (1 - y_matrix).*log(1 - h))) + regularization;
%printf("end of calculation J\n");


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
for t = 1:m
  a_1 = X(t,:);
  %printf("Size of a_1: %d %d\n", size(a_1, 1), size(a_1, 2));
  z_2 = a_1 * Theta1';
  %printf("Size of z_2: %d %d\n", size(z_2, 1), size(z_2, 2));
  a_2 = sigmoid(z_2);
  %printf("Size of a_2: %d %d\n", size(a_2, 1), size(a_2, 2));
  a_2 = [1 a_2];
  %printf("Size of a_2: %d %d\n", size(a_2, 1), size(a_2, 2));
  z_3 = a_2 * Theta2';
  %printf("Size of z_3: %d %d\n", size(z_3, 1), size(z_3, 2));
  a_3 = sigmoid(z_3);
  %printf("Size of a_3: %d %d\n", size(a_3, 1), size(a_3, 2));
  
  delta_3 = a_3 - y_matrix(t,:);
  %printf("Size of delta_3: %d %d\n", size(delta_3, 1), size(delta_3, 2));
  
  delta_2 = delta_3*Theta2;
  delta_2 = delta_2(2:end);
  delta_2 = delta_2 .* sigmoidGradient(z_2);
  %printf("Size of delta_2: %d %d\n", size(delta_2, 1), size(delta_2, 2));
  %printf("Size of delta_2: %d %d\n", size(delta_2, 1), size(delta_2, 2));
  
  Theta1_grad = Theta1_grad + delta_2'*a_1;
  Theta2_grad = Theta2_grad + delta_3'*a_2;
  
endfor

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
