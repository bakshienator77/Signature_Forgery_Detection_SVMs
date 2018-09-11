%Testing
function flag = Testing(file_name,path_name)
imshow([path_name,file_name]);
load('The_data_for_testing.mat');
    B=rgb2gray(imread([path_name,file_name]));
    B=imcrop(B, [180 180 1000 400]);
    %figure;
    %imagesc(B);
    B=imresize(B,0.1);
    %figure;
    %imagesc(B);
    X_test=B(:)';

X_testn=featureNormalize(double(X_test));
Z_test=projectData(X_testn, U, k);

p2 = svmPredict(model, Z_test);

if(p2==1)
fprintf('The signature is original');
else
    fprintf('The signature is a forgery')
end;
flag = p2;