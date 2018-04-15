load('a1TestData')
load('a1TrainingData')

% x =[-2.0000
%    -1.8000
%    -1.6000
%    -1.4000
%    -1.2000
%    -1.0000
%    -0.8000
%    -0.6000
%    -0.4000
%     0.4000
%     0.6000
%     0.8000
%     1.0000
%     1.2000
%     1.4000
%     1.6000
%     1.8000
%     2.0000];
% 
% y =[-19.6669
%   -15.7020
%   -13.6910
%   -12.4834
%   -12.6579
%   -13.0245
%    -9.3512
%   -12.2865
%   -11.4390
%    -6.3026
%    -9.5569
%    -8.5904
%    -5.8849
%    -7.3913
%    -3.5188
%    -4.6821
%     0.0424
%     1.7589];

cc = hsv(12);
r = [];
r_test = []
figure(1)
for k =1:3
    subplot(3,1,k)
    plot(x, y, 'o')
    hold on;
    w = polynomialRegression(x,y,k);
    y_hat = evalPolynomial(x,w);
    y_test_eval = evalPolynomial(xTest,w);
    L_test = norm(y_test_eval-yTest).^2;
    L = norm(y_hat - y).^2;
    x_range = -2.1:0.1:2.1;
    y_hat_plot = evalPolynomial(x_range, w);
    r = [r L];
    r_test = [r_test L_test];
    plot(x_range, y_hat_plot, 'color', cc(k,:));
    title(strcat('K = ', num2str(k)));
end
figure(2)
for k =4:6
    subplot(3,1,k - 3)
    plot(x, y, 'o')
    hold on;
    w = polynomialRegression(x,y,k);
    y_hat = evalPolynomial(x,w);
    y_test_eval = evalPolynomial(xTest,w);
    L_test = norm(y_test_eval-yTest).^2;
    L = norm(y_hat - y).^2;
    x_range = -2.1:0.1:2.1;
    y_hat_plot = evalPolynomial(x_range, w);
    r = [r L];
    r_test = [r_test L_test];
    plot(x_range, y_hat_plot, 'color', cc(k,:));
    title(strcat('K = ', num2str(k)));
end
figure(3)
for k =7:9
    subplot(3,1,k - 6)
    plot(x, y, 'o')
    hold on;
    w = polynomialRegression(x,y,k);
    y_hat = evalPolynomial(x,w);
    y_test_eval = evalPolynomial(xTest,w);
    L_test = norm(y_test_eval-yTest).^2;
    L = norm(y_hat - y).^2;
    x_range = -2.1:0.1:2.1;
    y_hat_plot = evalPolynomial(x_range, w);
    r = [r L];
    r_test = [r_test L_test];
    plot(x_range, y_hat_plot, 'color', cc(k,:));
    title(strcat('K = ', num2str(k)));
end

figure(4)
for k =10:12
    subplot(3,1,k - 9)
    plot(x, y, 'o')
    hold on;
    w = polynomialRegression(x,y,k);
    y_hat = evalPolynomial(x,w);
    y_test_eval = evalPolynomial(xTest,w);
    L_test = norm(y_test_eval-yTest).^2;
    L = norm(y_hat - y).^2;
    x_range = -2.1:0.1:2.1;
    y_hat_plot = evalPolynomial(x_range, w);
    r = [r L];
    r_test = [r_test L_test];
    plot(x_range, y_hat_plot, 'color', cc(k,:));
    title(strcat('K = ', num2str(k)));
end

figure(5)
k2 = [1:1:12];
plot(k2, r, '-b', k2, r_test, '-r');
legend('Training data','Test data');

% Polynomial of power 12 gives us the least residual error thus giving me
% the best fit for the training data. Polynomial of degree one has the
% highest error and thus such polynomial is of least use.

% Comparison of errors in training and test data:
% Both training and test data have decreasing error between k = 1 and k = 3
% Between k = 3 and k = 4, error in training and test data has a nearly
% constant behavior. Between k = 3 and k = 4, test data experiences an 
% upward kink.
% At k = 4, training data and test data error starts to diverge, 
% training data error continues to decrease and test data error starts
% have a substantial decrease. From lecture one, we know that overfitting
% happens when training data performs well but test data performs poorly.
% Therefore, by looking at the error graph, we can see that polynomial of
% degree 12 performs well with training data and poorly with test data.
% My hypothesis is that polynomial of degree 4 was used to generate the
% training and test data since both training and test data have a
% decreasing behavior before k = 4 and increasing behavior after k = 4; or,
% training data and test data start to diverge at k = 4. By the same rea-
% soning, we can say that the both models might have been based on k = 3 
% since at k = 3 both training and test data are equal.
% Therefore, both training data and test data might have been based on
% polynomial of either degree 3 or 4.






