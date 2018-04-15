row_1 = [];
for i=1:size(data_train,1)
    row_1 = [row_1 1];
end
data_train_transpose = data_train';
data_train_new = [row_1;data_train_transpose];

logistic_output = logisticReg(data_train_new, labels_train', 1); % output is 186 by 1. Our w's

w_sort_ascend = sort(logistic_output(:),'ascend');
w_sort_ascend = w_sort_ascend';
ham_numbers = w_sort_ascend(1,1:10);
spam_numbers = w_sort_ascend(1,(size(w_sort_ascend,2)-9):size(w_sort_ascend,2));

ham_index_vector = [];
for i=1:size(ham_numbers,2)
    index = find(logistic_output==ham_numbers(:,i),1);
    ham_index_vector = [ham_index_vector;index];
end

spam_index_vector = [];
for i=1:size(spam_numbers,2)
    index = find(logistic_output==spam_numbers(:,i),1);
    spam_index_vector = [spam_index_vector;index];
end
ham_index_vector'
spam_index_vector'

% v = 1
% ham_names = ['exe/zip/gif/jpg','grad','course','planning','2_parts','pjf','machine','thanks','finerty','statistical']
% spam_names = ['title','3_parts','reply','dinner','work','discuss','est','know','rate','sara']

