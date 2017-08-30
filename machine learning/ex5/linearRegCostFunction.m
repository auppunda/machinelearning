function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h_x = X * theta;

h_x = h_x - y;

h_x = h_x .* h_x;
J = sum(h_x(:));
J = J/(2*m);

Theta = theta .* theta;

J = J + lambda/(2*m) * (sum(Theta(:)) - Theta(1));


h = X * theta;

h = h - y;

h = X'*h;

grad = h/m;

Theta = theta;
Theta(1) = 0;

grad = grad + lambda/(m) * Theta;



% =========================================================================

grad = grad(:);

end
