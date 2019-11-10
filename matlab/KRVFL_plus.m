function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = KRVFL_plus(trainX,additionX, trainY,testX,testY, option)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an implementation of KRVFL+ with Matlab 2015b. 
% If you use it in your research, please cite the paper "Zhang Peng-Bo, and Yang Zhi-Xin. A new 
% learning paradigm for random vector functional-link network: RVFL+.Neural Networks 122 (2020) pp.94-105".
%% Parameters:
% trainX: Nsamples * Nfeatures matrix
% trainyY: Nsamples * 1 matrix(vector)
% addtionX: Nsamples * N-additional-features matrix
% testX: Msamples * Nfeatures matrix
% testY: Msamples * 1 matrix(vector)
% option: 
%   option.Type: 1 classification 0 regression
%   option.Kernel_class: 'RBF_kernel', 'lin_kernel', 'poly_kernel', 'wav_kernel'
%   option.Kernel_width
%   option.C
%   option.gamma
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(option,'Type')|| isempty(option.Type)
    option.Type = 1;
  
end
if ~isfield(option,'Kernel_class')|| isempty(option.Kernel_class)
    option.Kernel_class='RBF_kernel';
   
end
if ~isfield(option,'Kernel_width')|| isempty(option.Kernel_width)
    option.Kernel_width=0.02;
   
end
if ~isfield(option,'C')|| isempty(option.C)
    option.C=0.1;
   
end
if ~isfield(option,'gamma')|| isempty(option.gamma)
    option.gamma=1000;
   
end

%% Load data
gamma = option.gamma;
C = option.C;
Kernel_width = option.Kernel_width;
Kernel_class = option.Kernel_class;

%%%%%%%%%%% Load training dataset
T=trainY';
P=trainX';

%%%%%%%%%%% Load additional information
PF = additionX';

%%%%%%%%%%% Load testing dataset
TV.T = testY';
TV.P = testX';

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);

%% Convert labels into one-hot vectors 
if option.Type == 1
	U_trainY=unique(trainY);
	nclass=numel(U_trainY);
	trainY_temp=zeros(nclass, numel(trainY));
	for i=1:nclass
         idx= trainY==U_trainY(i);
        
         trainY_temp(i, idx)=1;
	end
	T = trainY_temp;
end
%% Training using training data and privileged information

tic;

%Construct omega_1 and omega_2

Omega_train_1 = kernel_matrix(P','lin_kernel');
Omega_train_2 = kernel_matrix(P',Kernel_class, Kernel_width);
Omega_train = Omega_train_1 + Omega_train_2;

%Construct tilde_omega_1 and tilde_omega_2

Omega_trainPF_1 = kernel_matrix(PF', 'lin_kernel',Kernel_width);
Omega_trainPF_2 = kernel_matrix(PF', Kernel_class, Kernel_width);
Omega_trainPF = Omega_trainPF_1+ Omega_trainPF_2;

%Calculate the output weights 

OutputWeight=(Omega_train+1/gamma*Omega_trainPF+eye(size(T,2))/C)\(T-C/gamma*ones(size(T))*Omega_trainPF)'; 
TrainingTime=toc;

%Calculate the training output
Y=(Omega_train * OutputWeight)'; 

%% Test using test data without privileged information

tic;
Omega_test_1 = kernel_matrix(P','lin_kernel',Kernel_width,TV.P');
Omega_test_2 = kernel_matrix(P',Kernel_class, Kernel_width,TV.P');
Omega_test = Omega_test_1 + Omega_test_2;
TY = (Omega_test' * OutputWeight)';                           
TestingTime=toc;
%% Calculate training and test accuracy

%regression case
if option.Type == 0
    TrainingAccuracy=sqrt(mse(T - Y));
    TestingAccuracy=sqrt(mse(TV.T - TY));           

%classification case
else
Y_temp=zeros(1, NumberofTrainingData);
Yt_temp=zeros(1,numel(TV.T));
for i=1:NumberofTrainingData
    [~,idx]=max(Y(:,i));
    Y_temp(i)=U_trainY(idx);
end
for i=1:numel(testY)
    [~,idx]=max(TY(:,i));
    Yt_temp(i)=U_trainY(idx);
end
TrainingAccuracy=length(find(Y_temp==trainY'))/NumberofTrainingData;

TestingAccuracy=length(find(Yt_temp==testY'))/NumberofTestingData;  
end


function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

nb_data = size(Xtrain,1);


if strcmp(kernel_type,'RBF_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./kernel_pars(1));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./kernel_pars(1));
    end
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<4,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xt');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
        XXh22 = sum(Xt,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
end