function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

predictions =  sigmoid(X*theta);

leftPart = -y' * log(predictions);

rightPart = (1 - y') * log(1 - predictions);

temp=theta;
temp(1)=0;
alpha=0.01;
updates = X' * (predictions - y);
lambaCostPart = (lambda / (2 * m)) * sum(temp .^ 2);
lambdaGradPart = lambda / m * temp;
J = (1 / m) * (leftPart - rightPart) + lambaCostPart;

grad = ((1/m) * (X' * (predictions - y)));% (unregularized)

grad = grad + lambdaGradPart;
   

grad = grad(:);

end
