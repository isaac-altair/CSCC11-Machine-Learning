error_vector = [];
error_number = 0;
for small_alpha =0:0.001:0.1
    naive_bayes = banana_alpha_dog(data_test,data_train,labels_train,small_alpha,1);
    class_naive_bayes = [];
    for i = 1:size(naive_bayes,1) % Compares two column vectors, find max, record the class of that max
        % Naive Bayes is 4000 x 2 matrix with class 0 in column 1 and class 1 in column 2
        row_naive_bayes = naive_bayes(i,:);
        max_probability = max(row_naive_bayes);
        if row_naive_bayes(1,1)==max_probability;
            class_naive_bayes = [class_naive_bayes 0];
        else
            class_naive_bayes = [class_naive_bayes 1];
        end        
    end
%     comparison_matrix = [class_naive_bayes;(labels_test)'];
%     for j = 1:size(comparison_matrix,2)
%         column_of_comparison_matrix = comparison_matrix(:,j);
%         if column_of_comparison_matrix(1,:)==column_of_comparison_matrix(2,:);
%             error_number = error_number + 0;
%         else
%             error_number = error_number + 1;
%         end
%     end
    error_number = numel(find(class_naive_bayes~=(labels_test)'));
    error_vector = [error_vector error_number];
    error_number = 0;
end
error_vector;

figure()
x_axis = 0:0.001:0.1; % 101 values
y_axis = error_vector;
plot(x_axis,y_axis,'-')
title('Error plot between test_labels and Naive Bayes')
xlabel('Alpha values')
ylabel('Error values')
legend('Error plot; beta = 1; alpha = 0:0.001:1')