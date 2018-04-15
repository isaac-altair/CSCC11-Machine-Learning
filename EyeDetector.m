% Simple Eigen-eyes detector. Load data (thanks to Francisco Estada  
% and Allan Jepson for allowing us to use this dataset).

load trainSet
load testSet

% the variables defined in the .mat files are:
% eyeIm - a 500 x n array, each COLUMN contains a vector that
%         represents an eye image
% nonIm - a 500 x m array, each COLUMN contains a vector that
%	  represents a non-eye image
% sizeIm - size of the eye and non eye images [y x]
who

% Normalize brightness to [0 1]
eyeIm=eyeIm/255;
nonIm=nonIm/255;
testEyeIm=testEyeIm/255;
testNonIm=testNonIm/255;

% You can display images from eyeIm or nonIm using;
%
% imagesc(reshape(eyeIm(:,1),sizeIm));axis image;colormap(gray)
%  - where of course you would select any column

% We will first see how far we can get with classification
% on the original data using kNN. The task is to distinguish
% eyes from non-eyes. This is useful to gain insight about
% how hard this problem is, and how much we can improve
% or lose by doing dimensionality reduction.

% Generate training and testing sets with classes for kNN,
% we need eye images to be on ROWS, not COLUMNS, and we also 
% need a vector with class labels for each

trainSet=[eyeIm'
          nonIm'];
trainClass=[zeros(size(eyeIm,2),1)
            ones(size(nonIm,2),1)];

testSet=[testEyeIm'
         testNonIm'];
testClass=[zeros(size(testEyeIm,2),1)
            ones(size(testNonIm,2),1)];

% Compute matrix of pairwise distances (this takes a while...)
d=som_eucdist2(testSet,trainSet);

% Compute kNN results, I simply chose a reasonable value
% for K but feel free to change it and play with it...
K=5;
[C,P]=knn(d,trainClass,K);

% Compute the class from C (we have 0s and 1s so it is easy)
class = sum(C,2);	  		% Add how many 1s there are
class = (class>(K/2));   % Set to 1 if there are more than K/2
				        % ones. Otherwise it's zero

% Compute classification accuracy: We're interested in 2 numbers:
% Correct classification rate - how many eyes were classified as eyes
% False-positive rate: how many non-eyes were classified as eyes

fprintf(2,'Correct classification rate:\n');
correctEye_knn=length(find(class(1:size(testEyeIm,2))==0))/size(testEyeIm,2)
fprintf(2,'False positive rate:\n');
falseEye_knn=length(find(class(size(testEyeIm,2)+1:end)==0))/size(testNonIm,2)

% Keep in mind the above figures! (and the kNN process, you'll
% have to do it again on the dimension-reduced data later on.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% PCA PART: Your task begins here!
% Do PCA on eyes and non-eyes to generate models for recognition


%%% TO DO:
% First, compute the mean over the eye and non-eye images
% i.e. compute the mean eye image, and the mean non-eye image
mean_eye_train = (mean(eyeIm.').'); % 500x1
assert(isequal(size(mean_eye_train), [500 1]))
mean_n_eye_train = (mean(nonIm.').');
assert(isequal(size(mean_n_eye_train), [500 1]))

%%% TO PRINT:
% Plot the mean eye and mean non-eye images and hand the
% printouts in with your report.

% Print eye images
imagesc(reshape(mean_eye_train(:,1),sizeIm));
axis image;
colormap(gray);

% Print non-eye images
imagesc(reshape(mean_n_eye_train(:,1),sizeIm));
axis image;
colormap(gray);

%%% TO DO:
% Now, perform the PCA computation as discussed in lecture over the 
% training set. You will do this separately for eye images and non-eye 
% images. This will produce a set of eigenvectors that represent eye 
% images, and a different set of eigenvectors for non-eye images.

sub_eyeIm = bsxfun(@minus, eyeIm, mean_eye_train);
assert(isequal(size(sub_eyeIm), [500 2392]))
sub_nonIm = bsxfun(@minus, nonIm, mean_n_eye_train);
assert(isequal(size(sub_nonIm), [500 1919]))

eye_cov = cov(sub_eyeIm');
assert(isequal(size(eye_cov), [500 500]))
non_eye_cov = cov(sub_nonIm');
assert(isequal(size(non_eye_cov), [500 500]))


% [V_eye, D_eye] = eig(eye_cov);
% assert(isequal(size(V_eye), [500 500]))
% assert(isequal(size(D_eye), [500 500]))
% [V_non_eye, D_non_eye] = eig(non_eye_cov);


%%% TO PRINT:
% Display and print out the first 5 eigenvectors for eyes and non-eyes 
% (i.e. the eigenvectors with LARGEST 5 eigenvalues, make sure you sort 
% the eigenvectors by eigenvalues!)

% eyes; need last 5 eigenvectors

% fix this

[V_e, D_e] = eig(eye_cov);
assert(isequal(size(V_e), [500 500]))
assert(isequal(size(D_e), [500 500]))
[V_n_e, D_n_e] = eig(non_eye_cov);
assert(isequal(size(eye_cov), [500 500]))
assert(isequal(size(eye_cov), [500 500]))


D_e = diag(D_e);
assert(isequal(size(D_e), [500 1]))
D_n_e = diag(D_n_e);
assert(isequal(size(D_n_e), [500 1]))

[sorted_D_e, index_e] = sort(D_e);
assert(isequal(size(sorted_D_e), [500 1]))
assert(isequal(size(index_e), [500 1]))

[sorted_D_n_e, index_n_e] = sort(D_n_e);
assert(isequal(size(sorted_D_n_e), [500 1]))
assert(isequal(size(index_n_e), [500 1]))


for n = 1:size(sorted_D_e, 1)
    sorted_V_e(:,n) = V_e(:, index_e(n));
end
assert(isequal(size(sorted_V_e), [500 500]))

for m = 1:size(sorted_D_n_e, 1)
    sorted_V_n_e(:,m) = V_n_e(:, index_n_e(m));
end  
assert(isequal(size(sorted_V_n_e), [500 500]))

sorted_V_e = fliplr(sorted_V_e);
assert(isequal(size(sorted_V_e), [500 500]))

sorted_V_n_e = fliplr(sorted_V_n_e);
assert(isequal(size(sorted_V_n_e), [500 500]))

D_e = (fliplr(D_e'))';
assert(isequal(size(D_e), [500 1]))

D_n_e = (fliplr(D_n_e'))';
assert(isequal(size(D_n_e), [500 1]))

% eyes; need first 5 eigenvectors

for q = 1:5
    figure(q)
    imagesc(reshape(sorted_V_e(:,q),sizeIm))
    axis image
    colormap(gray)
    pause(1)
end

% no eyes; need first 5 eigenvectors

hold on

for a = 6:10
    figure(a)
    imagesc(reshape(sorted_V_n_e(:,a-5),sizeIm))
    axis image
    colormap(gray)
    pause(1)
end
hold off

%%% TO DO:
% Now you have two PCA models: one for eyes, and one for non-eyes. 
% Next we will project our TEST data onto our eigenmodels to obtain 
% a low-dimensional representation first we choose a number of PCA
% basis vectors (eigenvectors) to use:

PCAcomp = 50;	% Choose 10 to start with, but you will 
		% experiment with different values of 
		% this parameter and see how things work
        
eyeVec = sorted_V_e(:,1:PCAcomp); % train data
assert(isequal(size(eyeVec), [500 PCAcomp]))

n_eyeVec = sorted_V_n_e(:,1:PCAcomp); % train data
assert(isequal(size(n_eyeVec), [500 PCAcomp]))

% To compute the low-dimensional representation for a given
% entry in the test test, we must do 2 things. First, we subtract
% the mean, and then we project that vector on the transpose of 
% the PCA basis vectors.  For example, say you have an eye image
%
% vEye=testSet(1,:);   % This is a 1x500 row vector
%
% The projections onto the PCA eigenvectors are:
%
% coeffEye=eyeVec(:,1:PCAcomp)'*(vEye'-eyeMean);
% coeffNonEye=noneyeVec(:,1:PCAcomp)'*(vNonEye'-noneyeMean);
%
% You need to compute coefficients for BOTH the eye and non-eye 
% models for each testSet entry, i.e. for each testSet image you 
% will end up with (2*PCAcomp) coefficients which are the projection 
% of that test image onto the chosen eigenvectors for eyes and non-eyes.

eyeMean_test = (mean(testEyeIm.').');
assert(isequal(size(eyeMean_test), [500 1]))

noneyeMean_test = (mean(testNonIm.').');
assert(isequal(size(noneyeMean_test), [500 1]))

test_vEye = testEyeIm';
assert(isequal(size(test_vEye), [2392 500]))

test_vNonEye = testNonIm';
assert(isequal(size(test_vNonEye), [1920 500]))

mat_coef_e = [];
for i = 1:size(test_vEye,1) % go through each row of vEye
    sub_vEye = test_vEye(i,:);
    test_coeffEye=eyeVec(:,1:PCAcomp)'*(sub_vEye'-eyeMean_test); 
    mat_coef_e = [mat_coef_e test_coeffEye];
end
test_mat_coef_e = mat_coef_e';

mat_coef_n_e = [];
for j = 1:size(test_vNonEye,1)
    sub_vNoneye = test_vNonEye(j,:);
    test_coeffNonEye=n_eyeVec(:,1:PCAcomp)'*(sub_vNoneye'-noneyeMean_test);
    mat_coef_n_e = [mat_coef_n_e test_coeffNonEye];
end
test_mat_coef_n_e = mat_coef_n_e';

% coeffEye=eyeVec(:,1:PCAcomp)'*(vEye'-eyeMean); 
% coeffNonEye=noneyeVec(:,1:PCAcomp)'*(vNonEye'-noneyeMean);
%
% Since we are going to use the KNN classifier demonstrated above, 
% you might want to place all the of the test coefficients into one 
% matrix.  You would then end up with a matrix that has one ROW for 
% each image in the testSet, and (2*PCAcomp) COLUMNS, one for each 
% of the coefficients we computed above.


%%% TO DO:
% Then do the same for the training data.  That is, compute the 
% PCA coefficients for each training image using both of the models.
% Then you will have low-dimensional test data and training data
% ready for the application of KNN, just as we had in the KNN example
% at the beginning of this script.

% mean_eye = (mean(eyeIm.').');
% mean_non_eye = (mean(nonIm.').');

% vEye = testEyeIm';
% vNonEye = testEyeIm';
train_vEye = eyeIm';
train_vNonEye = nonIm';

train_mat_coef_e = [];
for i = 1:size(train_vEye,1) % go through each row of vEye; 2392 iterations
    sub_train_vEye = train_vEye(i,:);
    train_coeffEye=eyeVec(:,1:PCAcomp)'*(sub_train_vEye'-mean_eye_train); % 10x500 * 500x1
    train_mat_coef_e = [train_mat_coef_e train_coeffEye];
end
train_mat_coef_e = train_mat_coef_e';

train_mat_coef_n_e = [];
for j = 1:size(train_vNonEye,1)  % go through each row of vNonEye; 1919 iterations
    sub_train_vNoneye = test_vNonEye(j,:);
    train_coeffNonEye=n_eyeVec(:,1:PCAcomp)'*(sub_train_vNoneye'-noneyeMean_test);
    train_mat_coef_n_e = [train_mat_coef_n_e train_coeffNonEye];
end
train_mat_coef_n_e = train_mat_coef_n_e';


%%% TO DO
% KNN classification: 
% Repeat the procedure at the beginning of this script, except
% instead of using the original testSet data, use the 
% coefficients for the training and testing data, and the same
% class labels for the training data that we had before
%
new_trainSet=[train_mat_coef_e
          train_mat_coef_n_e];
new_trainClass=[zeros(size(train_mat_coef_e',2),1)
            ones(size(train_mat_coef_n_e',2),1)];

new_testSet=[test_mat_coef_e
         test_mat_coef_n_e];
new_testClass=[zeros(size(test_mat_coef_e',2),1)
            ones(size(test_mat_coef_n_e',2),1)];

d_new = som_eucdist2(new_testSet,new_trainSet);

% Compute kNN results, I simply chose a reasonable value
% for K but feel free to change it and play with it...
% K = 5;
[C_new,P_new]=knn(d_new,new_trainClass,K);

% Compute the class from C (we have 0s and 1s so it is easy)
class_new = sum(C_new,2);	  		% Add how many 1s there are
class_new = (class_new>(K/2));   % Set to 1 if there are more than K/2
				        % ones. Otherwise it's zero

% Compute classification accuracy: We're interested in 2 numbers:
% Correct classification rate - how many eyes were classified as eyes
% False-positive rate: how many non-eyes were classified as eyes

% fprintf(2,'Correct classification rate for low-dimensional data:\n');
% correctEye_ldd =length(find(class_new(1:size(test_mat_coef_e',2))==0))/size(test_mat_coef_e',2)
% fprintf(2,'False positive rate for low-dimensional data:\n');
% falseEye_ldd =length(find(class_new(size(test_mat_coef_e',2)+1:end)==0))/size(test_mat_coef_n_e',2)

% fprintf(2,'Correct classification rate:\n');
% correctEye_knn=length(find(class(1:size(testEyeIm,2))==0))/size(testEyeIm,2)
% fprintf(2,'False positive rate:\n');
% falseEye_knn=length(find(class(size(testEyeIm,2)+1:end)==0))/size(testNonIm,2)

% HDD = correctEye_knn - falseEye_knn
% LDD = correctEye_ldd - falseEye_ldd



%%% TO PRINT:
% Print the classification accuracy and false-positive rates for the
% kNN classification on low-dimensional data and compare with the
% results on high-dimensional data.
%
% Discuss in your report: 
% - Are the results better? worse? is this what you expected?
% - why do you think the results are like this?
%

%%% TO DO:
% Finally, we will do classification directly from the PCA models
% for eyes and non-eyes.
%
% The idea is simple: Reconstruct each entry in the testSet
% using the PCA model for eyes, and separately the PCA model
% for non-eyes. Compute the squared error between the original
% entry and the reconstructed versions, and select the class
% for which the reconstruction error is smaller. It is assumed
% that the PCA model for eyes will do a better job of
% reconstructing eyes and the PCA model for non-eyes will
% do a better job for non-eyes (but keep this in mind:
% there's much more stuff out there that is not an eye
% than there are eyes!)
%
% To do the reconstruction, let's look at a vector from the
% coefficients we computed earlier for the training set;
%
% Reconstruction
%
% vRecon_eye= eyeMean + sum_k (eye_coeff_k * eye_PCA_vector_k);
%
% i.e. the mean eye image, plus the sum of each PCA component 
% multiplied by the corresponding coefficient. One can also replace
% the sum with a matrix-vector product.  Note: If you don't add 
% the mean image component back this won't work!
%
% Likewise, for the reconstruction using the non-eye model
%
% vRecon_noneye= nonMean + sum_k (noneye_coeff_k * noneye_PCA_vector_k)
%

% vRecon_eye_train = mean_eye + eyeVec * test_mat_coef_e';
% vRecon_n_eye_train = ;

recon_eye_mat_test = eyeVec * test_mat_coef_e';
vRecon_eye_test = bsxfun(@plus, recon_eye_mat_test, eyeMean_test); % gives the least
                                        % amount of error

recon_n_eye_mat_test = n_eyeVec * test_mat_coef_n_e';
vRecon_noneye_test = bsxfun(@plus, recon_n_eye_mat_test, noneyeMean_test);

%%% TO DO:
%
% Compute the reconstruction for each entry using the PCA model for eyes
% and separately for non-eyes, compute the error between these 2 
% reconstructions and the original testSet entry, and select the class
% that yields the smallest error.
%

D_eye_test = abs(testEyeIm - vRecon_eye_test).^2;
D_n_eye_test = abs(testNonIm - vRecon_noneye_test).^2; 

MSE_eye_test = sum(D_eye_test(:))/numel(testEyeIm);
MSE_n_eye_test = sum(D_n_eye_test(:))/numel(testNonIm);

% the eye class gives the least error

% calculate the correct and false classification rates for the PCA
% classifier

fprintf(2,'Correct classification rate for low-dimensional data:\n');
correctEye_ldd =length(find(class_new(1:size(test_mat_coef_e',2))==0))/size(test_mat_coef_e',2)
fprintf(2,'False positive rate for low-dimensional data:\n');
falseEye_ldd =length(find(class_new(size(test_mat_coef_e',2)+1:end)==0))/size(test_mat_coef_n_e',2)

fprintf(2,'Correct classification rate for PCA data:\n');
correctEye_PCA = length(find(class(1:size(vRecon_eye_test,2))==0))/size(vRecon_eye_test,2)
fprintf(2,'False positive rate for PCA data:\n');
falseEye_PCA = length(find(class(size(vRecon_eye_test,2)+1:end)==0))/size(vRecon_noneye_test,2)

LDD = correctEye_ldd - falseEye_ldd;
PCA = correctEye_PCA - falseEye_PCA;

%%% TO PRINT:
%
% Print the correct classification rate and false positive rate for
% the PCA based classifier and the low-dimensional kNN classifier
% using PCAcomps=5,10,15,25, and 50

% K = 5

%%% PCAcomp = 5
% correctEye_ldd =
% 
%     0.8353
% 
% False positive rate for low-dimensional data:
% 
% falseEye_ldd =
% 
%     0.1583
% 
% Correct classification rate for PCA data:
% 
% correctEye_PCA =
% 
%     0.9766
% 
% False positive rate for PCA data:
% 
% falseEye_PCA =
% 
%     0.1099

%%% PCAcomp = 10
% Correct classification rate:
% 
% correctEye_knn =
% 
%     0.9766
% 
% False positive rate:
% 
% falseEye_knn =
% 
%     0.1073
% 
% Correct classification rate for low-dimensional data:
% 
% correctEye_ldd =
% 
%     0.8737
% 
% False positive rate for low-dimensional data:
% 
% falseEye_ldd =
% 
%     0.1552
% 
% Correct classification rate for PCA data:
% 
% correctEye_PCA =
% 
%     0.9766
% 
% False positive rate for PCA data:
% 
% falseEye_PCA =
% 
%     0.1073

%%% PCAcomp = 15
% Correct classification rate:
% 
% correctEye_knn =
% 
%     0.9774
% 
% False positive rate:
% 
% falseEye_knn =
% 
%     0.1099
% 
% Correct classification rate for low-dimensional data:
% 
% correctEye_ldd =
% 
%     0.8725
% 
% False positive rate for low-dimensional data:
% 
% falseEye_ldd =
% 
%     0.1510
% 
% Correct classification rate for PCA data:
% 
% correctEye_PCA =
% 
%     0.9774
% 
% False positive rate for PCA data:
% 
% falseEye_PCA =
% 
%     0.1099

%%% PCAcomp = 25
% Correct classification rate:
% 
% correctEye_knn =
% 
%     0.9778
% 
% False positive rate:
% 
% falseEye_knn =
% 
%     0.1115
% 
% Correct classification rate for low-dimensional data:
% 
% correctEye_ldd =
% 
%     0.8737
% 
% False positive rate for low-dimensional data:
% 
% falseEye_ldd =
% 
%     0.1578
% 
% Correct classification rate for PCA data:
% 
% correctEye_PCA =
% 
%     0.9778
% 
% False positive rate for PCA data:
% 
% falseEye_PCA =
% 
%     0.1115

%%% PCAcomp = 50
% Correct classification rate:
% 
% correctEye_knn =
% 
%     0.9774
% 
% False positive rate:
% 
% falseEye_knn =
% 
%     0.1083
% 
% Correct classification rate for low-dimensional data:
% 
% correctEye_ldd =
% 
%     0.8495
% 
% False positive rate for low-dimensional data:
% 
% falseEye_ldd =
% 
%     0.1609
% 
% Correct classification rate for PCA data:
% 
% correctEye_PCA =
% 
%     0.9774
% 
% False positive rate for PCA data:
% 
% falseEye_PCA =
% 
%     0.1083


% Plot a graph of the kNN classification rate for the low-dimensional
% KNN classifier VS the number of PCA components (for the 5 values of 
% PCAcomps requested). 

PCAcomp_vec = [5 10 15 25 50]

correctEye_ldd_vec = [0.8737 0.8737 0.8725 0.8737 0.8495]

falseEye_ldd_vec = [0.1552 0.1552 0.1510 0.1578 0.1609];

correctEye_PCA_vec = [0.9766 0.9766 0.9774 0.9778 0.9774];

falseEye_PCA_vec = [0.1073 0.1073 0.1099 0.1115 0.1083];

diff_LDD = correctEye_ldd_vec - falseEye_ldd_vec
diff_PCA = correctEye_PCA_vec - falseEye_PCA_vec

% Plot of Low Dimensional Data

figure();
subplot(1,2,1)
plot(PCAcomp_vec, correctEye_ldd_vec, 'r');
hold on;
plot(PCAcomp_vec, falseEye_ldd_vec, 'b');
hold off;
title('Low-Dimensional KNN classifier:Generic2')
ylabel('Classification Values')
xlabel('Values of PCAcomp')
legend('correctEye_ldd_vec','falseEye_ldd_vec')
legend('location', 'best')
title('Low-Dimensional Classifier')

% figure();
subplot(1,2,2)
plot(PCAcomp_vec, correctEye_PCA_vec, 'r');
hold on;
plot(PCAcomp_vec, falseEye_PCA_vec, 'b');
hold off;
title('PCA classifier:Generic2')
ylabel('Classification Values')
xlabel('Values of PCAcomp')
legend('correctEye_PCA_vec','falseEye_PCA_vec')
legend('location', 'best')
title('PCA Classifier')

%
% Discuss in your Report:
% - Is there a value for PCAcomps (or set of values) for which low-dimensional
%   kNN is better than full dimensional kNN? 
% - why do you think that is?

% According to the graph, PCA components for which low-dimensional KNN per-
% forms better is located around PCAcomp = 15. Lower dimensional KNN does not 
% entail the curse of dimensionality which means it avoids ovefitting. Lower
% the PCAcomp, better the performance (as long as it makes false data small
% enough)

%
% Plot graphs of correct classification rate and the false-positive rate 
% fr the PCA-reconstruction classifier vs the number of PCA components.
%
% Discuss in your Report:
% - Which classifier gives the overall best performance?

% Overall, PCA performs better since the differnce between the correct and
% false values stays relatively high for all of the PCAcomps compared
% to the LDD. Although PCA has lower false values for PCAcomp at 5 and 50.

% - What conclusions can you draw about the usefulness of dimensionality
%   reduction?

% Lower dimensions help us describe and visualize data for the purpose of ob-
% serving patterns and making purposeful interpretation that would not be 
% possible in higher dimensions. Additionally, it speeds up the computational 
% process and removes redundant features. Overall, it helps us interpet data
% in a simpler and faster way. Additionally, it helps us avoid the curse of
% dimensionality.

% - Which classifier would you use on a large training set
%   consisting of high-dimensional data?

% PCA classifier

% - Which classifier would you use on a large training set
%   of low-dimensional (e.g. 3-D) data?
% - why?

% KNN Classifier

% We use PCA for high-dimensional data since PCA partitions high-dimensional
% data into smaller data sets which in turn enables us to make better analysis
% of the data by being able to compress, visualize, preprocess, and model the
% original data. Therefore, it provides the simplicity of analysis.
% We use KNN for low-dimensional data since we don't have many data points.
% Therefore, faster algorithm and we don't need dimensionality reduction. 

% - Summarize the advantages/disadvantages of each classifier!
%

% KNN:
% ---Disadvantages
% -Lazy learner
% -Can be slow if there is a great number of training points
% -Algorithm does not learn anything from the training data
% -Changing K changes predicted class labler
% -Expensive testing of each instance, as need to compute its distance to all
% known instances
% -Sensitivess to very unbalanced datasets
% -Changing K can change the predicted class label
% ---Advantages
% -Very simple classifier/implementation. Works well on basic recognition
% problems
% -Robust with regard to the search spaces; classifiers don't have to be li-
% nearyly separable
% -Few parameters to tune: distance metric and k
% -Can be updated easily as new instances with known classes are presented

% PCA:
% ---Disadvantages
% -Assumption that any relationships among the original attributes are essen-
% tially linear, or at least non-linear contribution is small
% -Results vary from whether data is normalized or not
% -Relies on orthogonal transformation
% -Values with greater variance gives greater importance
% ---Advantages
% -Acutal data can be described by a much lower-dimensional representation
% that caputres all of the structre of the data
% -Simplicity of the technique and its implementation
% -Can easily identify datasets which are responsible for the greatest variance
% -Avoids obscurity of the large data set
% -Robustness of the least squares approach to approximating the covariance or
% correlation matrix
% -Helps us visualize, preprocess, model, and compress data for indepth and
% purposeful analysis