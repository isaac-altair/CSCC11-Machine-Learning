function class = gccClassify(x, p1, m1, m2, C1, C2)
% 
% Inputs
%   x - test examplar
%   p1 - prior probability for class 1
%   m1 - mean of Gaussian measurement likelihood for class 1
%   m2 - mean of Gaussian measurement likelihood for class 2
%   C1 - covariance of Gaussian measurement likelihood for class 1
%   C2 - covariance of Gaussian measurement likelihood for class 2
%
% Outputs
%   class - sgn(a(x)) (ie sign of decision function a(x))

a(x) = (-0.5)*(x-m1)'*inv(C1)*(x-m1)+0.5*(x-m2)'*inv(C2)*(x-m2)+log(p1)-log(1-p1);
if a(x) < 0
    class = 0;
else
    class = 1;
end
    

% YOUR CODE GOES HERE.

