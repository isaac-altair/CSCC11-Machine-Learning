function class = knnClassify(test, k, trainingInputs, trainingTargets)
%
% Inputs:
%   test: test input vector
%   k: number of nearest neighbours to use in classification.
%   traingingInputs: array of training exemplars, one exemplar per row
%   traingingTargets: idenicator vector per row
%
% Basic Algorithm of kNN Classification
% 1) find distance from test input to each training exemplar,
% 2) sort distances
% 3) take smallest k distances, and use the mode class among 
%    those exemplars to label the test input.

classVector = [];

distanceVector = [];
for j = 1:size(trainingInputs,2);        % traverse through trainingInputs
    distanceVector = [distanceVector norm(abs(trainingInputs(:,j) - test(:,1)))]; % Populate distance vector with the difference
    % between each vector in trainingInputs and test vector
end
masterMatrix = [distanceVector;trainingTargets];
masterMatrixOrdered = (sortrows(masterMatrix',1))'; % Sort each column of the matrix wrt first row. Row 1 dictates the sorting
masterMatrixOrdered(end,:) = []; % Remove last row. 2 Rows remaining
masterMatrixOrdered(1,:) = []; % Remove first row. 1 row remaining, contains classes for specific vector i
if k <= length(masterMatrixOrdered);
    l = 1;
    while l <= k;
        classVector = [classVector masterMatrixOrdered(l)]; % Append k classes to classVector
        l = l + 1;
    end
end
class = mode(classVector);

