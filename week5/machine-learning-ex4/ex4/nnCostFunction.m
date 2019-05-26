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


% Forward Prop-------------------------------------------------
y_mat = zeros(size(y));
for i = 1:m
    y_mat(i, y(i)) = 1;
end
a1 = [ones(m, 1) X]; % 5000*401
z2 = a1*Theta1'; % 5000*25
a2 = sigmoid(z2); % 5000*25
a2 = [ones(m, 1) a2]; % 5000*26 (add bias unit)
z3 = a2*Theta2'; % 5000*10
a3 = sigmoid(z3); % 5000*10
h = a3; % 5000*10
J = (1/m)*sum(sum((-y_mat).*log(h) - (1-y_mat).*log(1 - h)));

unreg_theta1 = Theta1(:, 2:end);
unreg_theta2 = Theta2(:, 2:end);
reg = lambda * (1/(2*m)) * (sum(sum(unreg_theta1.^2)) + sum(sum(unreg_theta2.^2)));
J = J + reg;

% Back Prop----------------------------------------------------
delta1 = zeros(size(Theta1)); % 25*401 wide
delta2 = zeros(size(Theta2)); % 10*26 wide
for t = 1:m
%     a1_t = a1(t, :); %1*401
%     a2_t = a2(t, :); %1*25
%     a3_t = a3(t, :); %1*10
%     y_t = y_mat(t, :); %1*10
%     
%     d3 = a3_t - y_t; %1*10
%     d2 = Theta2'*d3' .* sigmoidGradient([1; Theta1*a1_t']); %26*1
%     
%     delta1 = delta1 + d2(2:end)*a1_t;
% 	delta2 = delta2 + d3' * a2_t;
    
    
    
a3_t =  a3(t, :); % 1*10
y_t = y_mat(t, :); % 1*10  
d_3 = a3_t - y_t; % 1*10
     
z2_t = [1 z2(t,:)]'; % 26*1
d_2 = Theta2'*d_3' .* sigmoidGradient(z2_t); %26*1

delta1 = delta1 + d_2(2:end)*a1(t,:); % 25*401
delta2 = delta2 + d_3'*a2(t,:); % 10*26
end

reg1 = Theta1;
reg1(:,1) = 0;
reg2 = Theta2;
reg2(:,1) = 0;
Theta1_grad = 1/m * (delta1 + (lambda)*reg1);
Theta2_grad = 1/m * (delta2 + (lambda)*reg2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
