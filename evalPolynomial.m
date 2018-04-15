function[y]=evalPolynomial(x,w)
    l_w = length(w) - 1;
    b = [];
    for i = 0:l_w
        b(end + 1,:) = (x.^i);
    end
    y = b' * w;
end