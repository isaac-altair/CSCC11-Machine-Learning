error_vector = [];
row_1 = [];
for i=1:size(data_train,1)
    row_1 = [row_1 1];
end
row_2 = [];
for j=1:size(data_test,1)
    row_2 = [row_2 1];
end
data_test_transpose = data_test';
new_data_test = [row_2;data_test_transpose];
data_train_transpose = data_train';
data_train_new = [row_1;data_train_transpose];
for v = 0:0.001:0.1
    logistic_output = logisticReg(data_train_new, labels_train', v); % output is 186 by 1
    result = logistic(new_data_test, logistic_output);
    result_round = round(result);
    error_number = numel(find(result_round~=(labels_test)'));
    error_vector = [error_vector error_number];
    error_number = 0;
end
error_vector;

figure()
x_axis = 0:0.001:0.1; % 101 values
y_axis = error_vector;
plot(x_axis,y_axis,'-')
title('Error plot between test_labels and Logistic Regression')
xlabel('V values')
ylabel('Error values')
legend('Error plot; v = 0:0.001:1')