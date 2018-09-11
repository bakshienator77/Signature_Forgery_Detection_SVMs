%VisComm Project
%X=zeros(21,750*400);
clear;
m=70;
for i=1:m
    fl=int2str(i);
    fl=strcat(fl,'.jpg');
    A=rgb2gray(imread(fl));     %loading image coverting to grayscale
    A=imcrop(A, [180 180 1000 400]);      %cropping image
    %figure;
    imagesc(A);
    A=imresize(A,0.1);        %reducing resolution to 10%
    %figure;
     imagesc(A);
    X(i,:)=A(:)';          %saving image
    if i<13
        y(i)=1;                         %assigning forgery/original status
    elseif i<29 
        y(i)=0;
    elseif i<35
        y(i)=1;
    elseif i<50
        y(i)=0;
    elseif i<52
       y(i)=1;
    elseif i<58
        y(i)=0;
    elseif i<67
        y(i)=1;
    else y(i)=0;
    end;
end;
y=y';
X_norm=featureNormalize(double(X));         %normalising the data
[U , S] =pca(X_norm);                   %applying PCA
Ssum=0;
for i=1:m
    Ssum=S(i,i)+Ssum;
end;
Dumvar=0;
for i=1:m
    Dumvar=S(i,i)+Dumvar;
    if (Dumvar/Ssum)>0.99
        k=i;                %Choosing k for 99% variance retained
        break;
    end;
end;

Z=projectData(X_norm, U, k);        %compressing data
Z_data=[Z y];                       %appending origianl/forgery status
Z_train(1,:)=Z_data(34,:);          %selecting a few examples as cross validation
Z_train(2,:)=Z_data(21,:);
Z_train(3,:)=Z_data(51,:);
Z_train(4,:)=Z_data(57,:);
Z_data(57,:)=[];
Z_data(51,:)=[];                    %removing said examples from training data
Z_data(34,:)=[];
Z_data(21,:)=[];
Z_data = Z_data(randperm(size(Z_data,1)),:);        %Randomising order of data
X=Z_data(:,1:k);
y=Z_data(:,k+1);

Xval=Z_train(:,1:k);
yval=Z_train(:,k+1);
error=zeros(8,100);
C=0.003;
sigma=1;
minerr=1;
for i=1:8                          %selecting ideal values for C and sigma
    C=(C + (0.007*(10)^((i-1)/2))*(mod(i,2)))*(1+2*mod(i+1,2));
    for j=1:100
        sigma=(sigma + (0.01*j));
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions=svmPredict(model,Xval);
        error(i,j)=mean(double(predictions~=yval));
        if(error(i,j)<=minerr)
            minerr=error(i,j);
            C_ans=C;
            sig_ans=sigma;
        end;
    end;
    sigma=0.003;
end;
C=C_ans;
sigma=sig_ans;
model = svmTrain(X,y,C,@(x1,x2) gaussianKernel(x1,x2,sigma));       %training algorithm to identify originals
p = svmPredict(model, X);               %predict over training set

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

p1 = svmPredict(model, Z_train(:,1:k));     %predicting over Cross validation set

fprintf('CV accuracy: %f\n', mean(double(p1==Z_train(:,k+1)))*100);
