function[master_posterior]=banana_alpha_dog(data_test,data_train,labels_train,small_alpha,small_beta)
class_0_matrix = [];
class_0_matrix = [class_0_matrix data_train(find(labels_train==0),:)];
class_1_matrix = [];
class_1_matrix = [class_1_matrix;data_train(find(labels_train==1),:)];
total_email_number = size(data_train,1); % row size of data_train
b_0 = (size(class_0_matrix,1)+small_beta)/(total_email_number+2*small_beta);
b_1 = (size(class_1_matrix,1)+small_beta)/(total_email_number+2*small_beta);
% Now, we need to work with alphas. Each column of a class is a feature.
% Calculate the number of k's in corresponding class matrix k
a_class_0_column = [];
% Traverse through the columns of class 0 and class 1
% class 0 traversion
for i = 1:size(class_0_matrix, 2); % traverse through 185 columns of class_0 matrix
    occurence_of_0 = sum(class_0_matrix(:,i)==1); % count the number of 0's in each column
    a_feature_class_0 = (occurence_of_0+small_alpha)/(size(data_train(labels_train==0,:),1)+2*small_alpha);
    a_class_0_column = [a_class_0_column;a_feature_class_0]; % column vector
end
a_class_0_column;
% class 1 traversion 
a_class_1_column = [];
for i = 1:size(class_1_matrix, 2);
    occurence_of_1 = sum(class_1_matrix(:,i)==1);
    a_feature_class_1 = (occurence_of_1+small_alpha)/(size(data_train(labels_train==1,:),1)+2*small_alpha);
    a_class_1_column = [a_class_1_column;a_feature_class_1]; % column vector
end

% [a_class_0_column a_class_1_column]
% pause;

% Extract the rows from data_test
alpha_0 = 0;
alpha_1 = 0;
alpha_0_output = [];
alpha_1_output = [];
a_class_0_row = a_class_0_column';
a_class_1_row = a_class_1_column';
for i = 1:size(data_test,1) % i goes from 1 to 4000
   row_data_test = data_test(i,:); % Extract each single row from data_test. row_1,2,...,4000
   alpha_0 = 0;
   % Now, traverse through each row to deal with alpha
   for k = 1:size(row_data_test,2); % Column 1 to 185
       if row_data_test(k)== 1; % if kth entry in row_data_test is 1
           alpha_0 = alpha_0 + log(a_class_0_row(k)); % take kth entry from class_0_column
       else
           alpha_0 = alpha_0 + log(1-a_class_0_row(k));
       end
   end
   alpha_0_output = [alpha_0_output;alpha_0 + log(b_0)];    
end

for i = 1:size(data_test,1); % i goes from 1 to 4000
   row_data_test = data_test(i,:); % Extract each single row from data_test. row_1,2,...,4000
   alpha_1 = 0;
   % Now, traverse through each row to deal with alpha
   for k = 1:size(row_data_test,2); % Column 1 to 185
       if row_data_test(k)== 1; % if kth entry in row_data_test
           alpha_1 = alpha_1 + log(a_class_1_row(k)); % take kth entry from class_0_column
       else
           alpha_1 = alpha_1 + log(1-a_class_1_row(k));
       end
   end
   alpha_1_output = [alpha_1_output;alpha_1+log(b_1)];
end

master_output = [alpha_0_output alpha_1_output];

gamma_vector = [];
for v=1:size(master_output,1)
    gamma = max(master_output(v,:));
    gamma_vector = [gamma_vector;gamma];
end

% 
% (alpha_0_output-gamma_vector)'
% (alpha_1_output-gamma_vector)'

denominator_c_0 = exp(alpha_0_output-gamma_vector);
denominator_c_1 = exp(alpha_1_output-gamma_vector);
master_denominator = denominator_c_0+denominator_c_1;

posterior_0 = [];
posterior_1 = [];
for l=1:size(alpha_0_output)
    posterior_0 = [posterior_0;exp((alpha_0_output(l)-gamma_vector(l)))];
end
posterior_0 = posterior_0./master_denominator;
master_posterior = [posterior_0];

for l=1:size(alpha_1_output)
    posterior_1 = [posterior_1;exp((alpha_1_output(l)-gamma_vector(l)))];
end
posterior_1 = posterior_1./master_denominator;

master_posterior = [master_posterior posterior_1];
end


