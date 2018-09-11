function [C, sigma] = dataset3Params(X, y, Xval, yval,x1,x2)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
error=zeros(8,8);
C=0.003;
sigma=0.003;
minerr=1;
for i=1:8
    C=(C + (0.007*(10)^((i-1)/2))*(mod(i,2)))*(1+2*mod(i+1,2));
    for j=1:8
        sigma=(sigma + (0.007*(10)^((j-1)/2))*(mod(j,2)))*(1+2*mod(j+1,2));
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions=svmPredict(model,Xval);
        error(i,j)=mean(double(predictions~=yval));
        if(error(i,j)<minerr)
            minerr=error(i,j);
            C_ans=C;
            sig_ans=sigma;
        end;
    end;
    sigma=0.003;
end;
C=C_ans;
sigma=sig_ans;

% =========================================================================

end
