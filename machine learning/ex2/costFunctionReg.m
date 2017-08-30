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

for iter = 1:m
    X_new = mapFeature(X(iter, 2), X(iter,3));
    x = 0;
    for it = 1:length(theta)
        x = x + theta(it)*X_new(1,it);
    end
    J = J - y(iter, 1)*log(sigmoid(x)) - (1 - y(iter,1))* log(1 - sigmoid(x));
    for s = 1 : length(theta)
       grad(s,1) = grad(s,1) + (sigmoid(x) - y(iter,1))*X_new(1,s); 
    end
end
J = J/m;
thetaC = 0;
grad = grad/m;

for it = 2:length(theta)
    thetaC = theta(it) * theta(it);
    grad(it,1) = grad(it,1) + theta(it) * lambda/m;
end
thetaC = thetaC*lambda/m;

J = J + thetaC;
% =============================================================
end
