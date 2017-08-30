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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_Vector = zeros(length(y), length(Theta2(:,1)));

for is = 1:m 
   y_Vector(is, y(is)) = 1; 
end
X =  [ones(length(X),1) , X];

a_2 = Theta1 * X';

a_2 = a_2';

z_2 = sigmoid(a_2);

z_2 = [ones(length(X),1), z_2];

a_3 = Theta2 * z_2';

a_3 = a_3';

z_3 = sigmoid(a_3);


sdt = -y_Vector.*log(z_3) - (1 - y_Vector).*log(1-z_3);


J = sum(sum(sdt));
J = J/m;

Theta1s = Theta1 .* Theta1;
Theta2s = Theta2 .* Theta2;

J = J + lambda/(2*m) * (sum(Theta1s(:)) + sum(Theta2s(:)) - sum(Theta1s(:,1)) -  sum(Theta2s(:,1)));

ThetaPrime = Theta1';
Theta2Prime = Theta2';

%delta_2 = Theta2Prime*delta_3';
%delta_2 = delta_2' .* sigmoidGradient(z_2);
%delta_2 = delta_2(1:end,2:end);
Triangle1 = zeros(size(Theta1_grad));
Triangle2 = zeros(size(Theta2_grad));
a_2 = [ones(length(X),1),a_2];
for i = 1:m
   delta_3 = z_3(i,:) - y_Vector(i,:);
   delta_2 = (Theta2' * delta_3')' .* sigmoidGradient(a_2(i,:)); 
   delta_2 = delta_2(1:end, 2:end);
   Triangle1 = Triangle1 + (delta_2' * X(i,:));
   
   Triangle2 = Triangle2 + (delta_3' * z_2(i,:));
    
end
    

%triangle_1 = delta_2' * X;

%triangle_1 = triangle_1/m;
Theta1_grad = Triangle1/m;
Theta2_grad = Triangle2/m;


regularized1 = lambda*Theta1/m;
regularized2 = lambda*Theta2/m;

for i = 1:length(Theta1(:,1)) 
   regularized1(i,1) = 0; 
    
end

for i = 1:length(Theta2(:,1)) 
   regularized2(i,1) = 0; 
    
end

Theta1_grad = Theta1_grad + regularized1;
Theta2_grad = Theta2_grad + regularized2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
