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

% Part 1
X = [ones(m,1),X];
a2 = sigmoid(X*Theta1');
a2 = [ones(m,1),a2];
h = sigmoid(a2*Theta2');

yv = zeros(m, num_labels);
for i = 1:m
	Y = zeros(10,1);
	Y(y(i)) = 1;
	yv(i, y(i)) = 1; % yv = vectorized form of y
	
	for k = 1:num_labels
		J = J + (-Y(k)*log(h(i,k))) - ((1 - Y(k))*log(1-h(i,k)));
	end
end

%Regularization
R = sum(sum(Theta1(:,[2:end]).^2)) + sum(sum(Theta2(:,[2:end]).^2));
R = (R*lambda) / (2*m);

J = J/m + R;

%Part 2
bigDelta_1 = 0;
bigDelta_2 = 0;

for t = 1:m
    
    %Fordward Propagation
    a1 = X(t,:); % dim = 1 x 401

    z2 = a1*Theta1'; % size(Theta1) = 25x401  size(z2)=1x25
    a2 = sigmoid(z2); % dim = 1x25
    a2 = [1 a2]; %dim = 1x26
    
    z3 = a2*Theta2'; %size(Theta2) = 10x26
    a3 = sigmoid(z3);  %size(Theta1)=10x26 size(a3)=1x10 
    
    %Deltas
    delta_3 = a3 - yv(t,:); %dim = 1x10

    delta_2 = (delta_3*Theta2) .* a2 .* (1 - a2);
    delta_2 = delta_2(2:end); %dim=1x25
    
    bigDelta_1 = bigDelta_1 + delta_2'*a1;
    bigDelta_2 = bigDelta_2 + delta_3'*a2;

end

Theta1_grad = bigDelta_1./m;
Theta2_grad = bigDelta_2./m;

%Part 3
Theta1_reg = (lambda/m)*Theta1;
Theta1_reg(:,1) = 0;

Theta2_reg = (lambda/m)*Theta2;
Theta2_reg(:,1) = 0;

Theta1_grad = Theta1_grad + Theta1_reg;
Theta2_grad = Theta2_grad + Theta2_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
