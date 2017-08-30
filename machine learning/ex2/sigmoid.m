function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


for iter = 1:size(z)
    for sze = 1:size(z,2)
    g(iter, sze) = 1/(1 + exp(-z(iter, sze)));
    end
end
% =============================================================

end
