function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];

for j=1:m,

	%first layer propagation
	z2 = sigmoid(X(j,:) * Theta1');

	%bias to hidden layer
	z2 = [1 z2];

	%hidden layer propagation and getting max (candidate)
	[trash,p(j)] = max(sigmoid(z2 * Theta2'));
end;


end
