function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
options = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
pairs = zeros(length(options)^2, 2);
l = 1;
for i = 1:length(options)
    for j = 1:length(options)
        pairs(l, :) = [options(i), options(j)];
        l = l+1;
    end
end

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
errors = zeros(length(pairs), 1);
for i = 1:length(pairs)
    C_i = pairs(i,1);
    sigma_i = pairs(i,2);
    
    model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i)); 
    predictions = svmPredict(model, Xval);
    errors(i) = mean(double(predictions ~= yval));
end

[min_cost, min_cost_index] = min(errors);
C = pairs(min_cost_index, 1);
sigma = pairs(min_cost_index, 2);


% =========================================================================

end
