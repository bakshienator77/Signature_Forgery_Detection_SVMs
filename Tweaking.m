clear;
load('Z dataset.mat');

Z=projectData(X_norm, U, k);
Z_data=[Z y];
Z_train(1,:)=Z_data(1,:);
Z_train(2,:)=Z_data(13,:);
Z_data(13,:)=[];
Z_data(1,:)=[];
Z_data = Z_data(randperm(size(Z_data,1)),:);
X=Z_data(:,1:k);
y=Z_data(:,k+1);

Xval=Z_train(:,1:k);
yval=Z_train(:,k+1);
error=zeros(8,100);
C=0.003;
sigma=1;
minerr=1;
for i=1:8
    C=(C + (0.007*(10)^((i-1)/2))*(mod(i,2)))*(1+2*mod(i+1,2));
    for j=1:100
        sigma=(sigma + (0.01*j));
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
model = svmTrain(X,y,C,@(x1,x2) gaussianKernel(x1,x2,sigma));
p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

p1 = svmPredict(model, Z_train(:,1:k));

fprintf('CV accuracy: %f\n', mean(double(p1==Z_train(:,k+1)))*100);
