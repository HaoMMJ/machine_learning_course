function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

zeta = X*theta;
h = sigmoid(zeta);
J = (1/m)*sum(-y.*log(h) - (1 - y).*log(1 - h)) + (lambda/(2*m))*(sum(theta.^2) - theta(1)^2); 

for i = 1:rows(grad)
  for j = 1:columns(grad)
    %printf("%f %f \n", i,j);
    %printf("gradient calculation %f abcdef\n", j);
    %disp(X(:,i));
    if(i == 1)
      grad(i,j) = (1/m)*sum((h - y).*X(:,i));
    else
      grad(i,j) = (1/m)*sum((h - y).*X(:,i)) + (lambda/m)*theta(i);
    endif
    %printf("grad %f\n", grad(i,j));
  endfor
endfor




% =============================================================

end
