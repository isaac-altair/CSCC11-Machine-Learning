a_class_0_vector=banana_alpha_dog(data_test,data_train,labels_train,1,1);
banana = sort(a_class_0_vector(:),'descend');
max_10 = banana(1:10);
index_vector = [];
for i=1:size(max_10,1)
    index = find(a_class_0_vector==max_10(i,:),1);
    index_vector = [index_vector;index];
end

max_10';
% vector = [a_class_0_vector(180,:),a_class_0_vector(177,:),a_class_0_vector(178,:),
  %a_class_0_vector(173,:),a_class_0_vector(154,:),a_class_0_vector(154,:),a_class_0_vector(175,:),a_class_0_vector(139,:),a_class_0_vector(160,:),a_class_0_vector(171,:)];
vector;
index_vector
%% 

a_class_1_vector=banana_alpha_dog(data_test,data_train,labels_train,1,1);
banana = sort(a_class_1_vector(:),'descend');
max_10 = banana(1:10);
index_vector = [];
for i=1:size(max_10,1)
    index = find(a_class_1_vector==max_10(i,:),1);
    index_vector = [index_vector;index];
end

max_10';
% vector = [a_class_0_vector(180,:),a_class_0_vector(177,:),a_class_0_vector(178,:),
 %a_class_0_vector(173,:),a_class_0_vector(154,:),a_class_0_vector(154,:),a_class_0_vector(175,:),a_class_0_vector(139,:),a_class_0_vector(160,:),a_class_0_vector(171,:)];
vector;
index_vector

%% 

% Below, we used small_alpha = 1 and small_beta = 1

% Top 10 Ham:
% ham_naive_bayes = ['no_multipart','toronto','sam','date','know','know','department','please','thanks','computer']

% Top 10 Spam:
% spam_naive_bayes = ['no_multipart','iso','alternative','quoted','priority','public','express','microsoft','3_parts','msmail']
