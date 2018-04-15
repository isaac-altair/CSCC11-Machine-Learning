function [ll, g] = logisticNLP(x1, x2, w, alpha)
% [ll, g] = logisticNLP(x1, x2, w, alpha);
% 
% Inputs:
%   x1 - array of exemplar measurement vectors for class 1.
%   x2 - array of exemplar measurement vectors for class 2.
%   w - an array of weights for the logistic regression model.
%   alpha - weight decay parameter
% Outputs:
%   ll - negative log probability (likelihood) for the data 
%        conditioned on the model (ie w).
%   g - gradient of negative log data likelihood
%       (partial derivatives of ll w.r.t. elements of w)

% yi can be either 0 or 1
% w is 2 by 1 vector
% we are looping over each column of x1 and x2

% sum = 0;
% j = 1; 
% while j < size(x1,2);
%     sum = sum + (-log(1+exp(-w'*x1(:,j)))-log(1+exp(-w'*x2(:,j)))+w'*x2(:,j));
%     j = j + 1;
% end
% 
% ll = -log(2*pi*alpha.^2)^(-size(w,2)/2)+(norm(w).^2)/(2*alpha)-sum;
% 
% l = 1
% while l<=size(x1,2);
%     sum = sum + (norm(x1(:,l),1)*(exp(-w'*x1(:,j)))/(1+exp(-w'*x1(:,j)))+norm(x2(:,l),1)*(exp(-w'*x2(:,l)))/(1+exp(-w'*x2(:,l)))+norm(x2(:,l),1));
%     l = l + 1
% end
% g = norm(w,1)/alpha + sum

sum = 0;
j = 1;

oneVector = [];
o = 1;
while o <= size(x1,2);
    oneVector = [oneVector 1];
    o = o + 1;
end

x1 = [x1;oneVector];
x2 = [x2;oneVector];

while j < size(x1,2);
    sum = sum + (log(1+exp(w'*x1(:,j)))+log(1+exp(w'*x2(:,j)))-w'*x1(:,j)-2*w'*x2(:,j));
    j = j + 1;
end

ll = -log(2*pi*alpha)^(-size(w,2)/2)+(norm(w).^2)/(2*alpha)+sum;

l = 1;
while l<=size(x1,2);
    sum = sum + ((norm(x1(:,l),1)*exp(w'*x1(:,j))/(1+exp(w'*x1(:,j)))+norm(x2(:,l),1)*exp(w'*x2(:,j))/(1+exp(w'*x2(:,l)))-norm(x1(:,l),1))-2*norm(x2(:,l),1));
    l = l + 1;
end
g = norm(w,1)/alpha + sum;

