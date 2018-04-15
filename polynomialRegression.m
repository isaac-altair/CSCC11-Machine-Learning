function[w]=polynomialRegression(x,y,k)
B = [];
for i=0:k
%     append(B, x.^i)
    B(end + 1,:) = (x.^i);
% b = B';
% w = inv(B'*B)*B'*y
w = B'\y;
w
end




        
        
