function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
C_pos = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
X1 = X(:, 1);
X2 = X(:, 2);
model = svmTrain(X, y, C_pos(1), @(x1, x2)gaussianKernel(x1, x2, C_pos(1)));
vec = svmPredict(model, Xval);
means_min = mean(double(vec ~= yval));
i_val = 1;
p_val = 1;
for i = 1:length(C_pos)
   for p = 1:length(C_pos) 
        model = svmTrain(X, y, C_pos(i), @(x1, x2)gaussianKernel(x1, x2, C_pos(p)));
        vec = svmPredict(model, Xval);
        means = mean(double(vec ~= yval));
        if means < means_min
           means_min = means;
           i_val = i;
           p_val = p;
        end
   end   
end

C = C_pos(i_val);
sigma = C_pos(p_val);


% =========================================================================

end
