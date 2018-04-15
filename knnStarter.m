
% Use the function knnClassify to test performance on different datasets.

load('fruit_test')
load('fruit_train')

masterClassVector = [];
for i = 1:size(inputs_test,2) % traverse through the columns of inputs_test
    classVector = [];
    for k = 3:2:21 % traverse through k = 3,5,7...,15,18,21. Take each column vector from inputs_test and apply different
                    % values of k. Record the class value for each k
       class = knnClassify(inputs_test(:,i), k, inputs_train, target_train); % gives 10 different classes for each column vector
                                        % of inputs_test given 10 different
                                        % k's
                                        % On each iteration, classVector
                                        % has to be populated by 10 entries
       classVector = [classVector class]; % Append class value for each k = 3,5,...,18,21  pertaining
                                         % to each specific column vector
    end 
    masterClassVector = [masterClassVector;classVector];
end

target_test(end,:) = [];
masterErrorArray = [];
for m = 1:size(masterClassVector,1) % get total number of rows
    % Now, want to compare row vector with target_test
    column_vector = masterClassVector(:,m); % get the row vector
    error_array = [];
    counter = 0;
    for n = 1:size(masterClassVector(m,:),2) % get total number of columns
        if column_vector(n)'~=target_test(n) %compare values in row vector with that of inputs_train
            counter = counter + 1;
        end
    end
    error_array = [error_array counter];
    masterErrorArray = [masterErrorArray error_array];
    counter = 0;
end
masterErrorArray;  %%% Y-Coordinate of the Error graph %%%

k_values = [];
for k = 3:2:21
    k_values = [k_values k];
end
k_values;

figure()
plot1 = plot(k_values, masterErrorArray, '--go')
title('Plot of Test Error:Fruits')
ylabel('Error values')
xlabel('Values of K')
legend('Error w.r.t k: Fruits')

hold on
%% 

load('mnist_test')
load('mnist_train')
masterClassVector = [];
target_test(end,:) = []; % get the first row of target_test
for i = 1:size(inputs_test,2); % traverse through the columns of inputs_test; inputs_test is 784x200. 200 columns
    classVector = [];
    for k = 3:2:21; % traverse through k = 3,5,7...,15,18,21. Take each column vector from inputs_test and apply different
                    % values of k. Record the class value for each k
       class = knnClassify(inputs_test(:,i), k, inputs_train, target_train); 
                                        % gives 10 different classes for
                                        % each column vector. column vector
                                        % is of size 700x1
                                        % On each iteration, classVector
                                        % has to be populated by 10
                                        % entries.
                                        % inputs_train is 784x60
                                        % target_train is 2x60
       classVector = [classVector class]; % Append class value for each k = 3,5,...,18,21  pertaining
                                         % to each specific column vector
                                         % size has to be 1x10
    end 
    masterClassVector = [masterClassVector;classVector]; % size has to be
                                                         % 200x10, since we
                                                         % have 200 columns
end

size(masterClassVector)

counter = 0;
% target_test(end,:) = [];
masterErrorArray = [];
for m = 1:size(masterClassVector,2) % get total number of columns. 10 rows
    % Now, want to compare row vector with target_test
    column_vector = masterClassVector(:,m); % get the column vector
    error_array = [];
    for n = 1:size(masterClassVector(m,:),2); % get total number of columns. 10 columns
        if column_vector(n)'~=target_test(n); %compare values in row vector with that of inputs_train
            counter = counter + 1;
        end
    end
    error_array = [error_array counter];
    masterErrorArray = [masterErrorArray error_array];
    counter = 0;
end
masterErrorArray;  %%% Y-Coordinate of the Error graph %%%

k_values = [];
for k = 3:2:21
    k_values = [k_values k];
end
k_values;

figure()
plot2 = plot(k_values, masterErrorArray, '--bo')
title('Plot of Test Error:Digits')
ylabel('Error values')
xlabel('Values of K')
legend('Error w.r.t k:MNist')

hold on
%%

load('generic1')
% load('generic2')
c_test_inputs = [c1_test c2_test]; % size 2x138
c_train_inputs = [c1_train c2_train]; % size 2x42
c_target_test = [ones(1,69) zeros(1,69)]; % size 1x138
c_target_train = [ones(1,21) zeros(1,21);zeros(1,21) ones(1,21)]; % size 2x42

masterClassVector = [];
for i = 1:size(c_test_inputs,2); % traverse through the columns of inputs_test; inputs_test is 784x200. 200 columns
    classVector = [];
    for k = 3:2:21; % traverse through k = 3,5,7...,15,18,21. Take each column vector from inputs_test and apply different
                    % values of k. Record the class value for each k
       class = knnClassify(c_test_inputs(:,i), k, c_train_inputs, c_target_train); 
                                        % gives 10 different classes for
                                        % each column vector.
       classVector = [classVector class]; % Append class value for each k = 3,5,...,18,21  pertaining
                                         % to each specific column vector
                                         % size has to be 1x10
    end 
    masterClassVector = [masterClassVector;classVector];
end

masterClassVector;

counter = 0;
masterErrorArray = [];
for m = 1:size(masterClassVector,2) % get total number of columns. 10 columns. 10 cylces.
    % Now, want to compare row vector with target_test
    column_vector = masterClassVector(:,m); % get the column vector
    error_array = [];
    for n = 1:size(masterClassVector(:,m),1); % Traverse through each 10 columns, each column contains 138 entries
        if column_vector(n)'~=c_target_test(n); %compare values in column vector with that of inputs_train
            counter = counter + 1;
        end
    end
    error_array = [error_array counter];
    masterErrorArray = [masterErrorArray error_array];
    counter = 0;
end
masterErrorArray;  %%% Y-Coordinate of the Error graph %%%

k_values = [];
for k = 3:2:21
    k_values = [k_values k];
end
k_values;

figure()
plot3 = plot(k_values, masterErrorArray, '--ro')
title('Plot of Test Error:Generic1')
ylabel('Error values')
xlabel('Values of K')
legend('Error w.r.t k:Generic1')

hold on
%% .

load('generic2')

c_test_inputs = [c1_test c2_test]; % size 2x138
c_train_inputs = [c1_train c2_train]; % size 2x42
c_target_test = [ones(1,69) zeros(1,69)]; % size 1x138
c_target_train = [ones(1,21) zeros(1,21);zeros(1,21) ones(1,21)]; % size 1x42

masterClassVector = [];
for i = 1:size(c_test_inputs,2); % traverse through the columns of inputs_test; inputs_test is 784x200. 200 columns
    classVector = [];
    for k = 3:2:21; % traverse through k = 3,5,7...,15,18,21. Take each column vector from inputs_test and apply different
                    % values of k. Record the class value for each k
       class = knnClassify(c_test_inputs(:,i), k, c_train_inputs, c_target_train); 
       classVector = [classVector class]; % Append class value for each k = 3,5,...,18,21  pertaining
                                         % to each specific column vector
                                         % size has to be 1x10
    end 
    masterClassVector = [masterClassVector;classVector]; 
end

masterClassVector;

counter = 0;
masterErrorArray = [];
for m = 1:size(masterClassVector,2) % get total number of columns. 10 rows
    % Now, want to compare row vector with target_test
    column_vector = masterClassVector(:,m); % get the column vector
    error_array = [];
    for n = 1:size(masterClassVector(:,m),1); % get total number of columns. 10 columns
        if column_vector(n)'~=c_target_test(n); %compare values in column vector with that of inputs_train
            counter = counter + 1;
        end
    end
    error_array = [error_array counter];
    masterErrorArray = [masterErrorArray error_array];
    counter = 0;
end
masterErrorArray;  %%% Y-Coordinate of the Error graph %%%

k_values = [];
for k = 3:2:21
    k_values = [k_values k];
end
k_values;

figure()
plot4 = plot(k_values, masterErrorArray, '--ko')
title('Plot of Test Error:Generic2')
ylabel('Error values')
xlabel('Values of K')
legend('Error w.r.t k:Generic2')
