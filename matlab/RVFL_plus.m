function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = RVFL_plus(trainX,additionX, trainY,testX,testY, option)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an implementation of RVFL+ with Matlab 2015b. 
% If you use it in your research, please cite the paper "Zhang Peng-Bo, and Yang Zhi-Xin. A new 
% learning paradigm for random vector functional-link network: RVFL+.Neural Networks 122 (2020) pp.94-105".
%% Parameters:
% trainX: Nsamples * Nfeatures matrix
% trainyY: Nsamples * 1 matrix(vector)
% addtionX: Nsamples * N-additional-features matrix
% testX: Msamples * Nfeatures matrix
% testY: Msamples * 1 matrix(vector)
% option: 
%   option.N: number of hidden nodes
%   option.seed: random seed
%   option.RandomType: 'Uniform' or 'Gaussian'
%   option.ActivationFunction: 'sigmoid'/'sig', 'sin'/'sin',
%     'hardlim','tribas', 'radbas'
%   option.ScaleValue: u
%   option.Type: 1 classification 0 regression
%   option.C
%   option.gamma
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfield(option,'N')|| isempty(option.N)
    option.N=100;
end
if ~isfield(option,'ActivationFunction')|| isempty(option.ActivationFunction)
    option.ActivationFunction='sigmoid';
end
if ~isfield(option,'seed')|| isempty(option.seed)
    option.seed=0;
end
if ~isfield(option,'RandomType')|| isempty(option.RandomType)
    option.RandomType='Uniform';
end
if ~isfield(option,'Type')|| isempty(option.Type)
    option.Type = 1;
  
end
if ~isfield(option,'ScaleValue')|| isempty(option.ScaleValue)
    option.ScaleValue=1;
   
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
u = option.ScaleValue;
NumberofHiddenNeurons = option.N;
ActivationFunction = option.ActivationFunction;
%%%%%%%%%%% Load training dataset
T=trainY';
P=trainX';

%%%%%%%%%%% Load additional information
PF = additionX';

%%%%%%%%%%% Load testing dataset
TV.T = testY';
TV.P = testX';

[NumberofInputNeurons, NumberofTrainingData] = size(P);
[NumberofPFInputNeurons, ~] = size(PF);
NumberofTestingData=size(TV.P,2);

rand('state',option.seed);
randn('state',option.seed);
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

%Randomly generate weights and biases between the input layer and the enhancement nodes
 if strcmp(option.RandomType,'Uniform') 
          Weight=u*(rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1);
          Bias=u*(rand(NumberofHiddenNeurons,1));
		  PFWeight=u*(rand(NumberofHiddenNeurons,NumberofPFInputNeurons)*2-1);
 else if strcmp(option.RandomType,'Gaussian')
           Weight=u*(randn(NumberofHiddenNeurons,NumberofInputNeurons)*2-1);
           Bias=u*(randn(NumberofHiddenNeurons,1));  
		   PFWeight=u*(randn(NumberofHiddenNeurons,NumberofPFInputNeurons)*2-1);
 else
       error('only Gaussian and Uniform are supported')
     end
 end 

BiasMatrix=repmat(Bias,1,NumberofTrainingData);
G=Weight*P+BiasMatrix;
GPF = PFWeight*PF + BiasMatrix;

% Calculate activation function
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        H = 1 ./ (1 + exp(-G));
        HPF = 1 ./ (1 + exp(-GPF));
    case {'sin','sine'}
        H = sin(G);
        HPF = sin(GPF);
    case {'hardlim'}
        H = double(hardlim(G));
        HPF = double(hardlim(GPF));
    case {'tribas'}
        H = tribas(G);
        HPF = tribas(GPF);
    case {'radbas'}
        H = radbas(G);
        HPF = radbas(GPF);               
end
                                       %   Release the temparary array
%Consider the direct link
H = [P;H];
HPF = [PF;HPF];
H(isnan(H)) = 0;
HPF(isnan(HPF)) = 0;
% Calculate the output weights

% OutputWeight = H*inv(H'*H+1/gamma*HPF'*HPF+speye(size(H,2))/C)*(T-C/gamma*ones(size(T))*HPF'*HPF)'; 
OutputWeight = H/(H'*H+ 1/gamma*HPF'*HPF+speye(size(H,2))/C)*(T-C/gamma*ones(size(T))*HPF'*HPF)';
TrainingTime=toc;

% Calculate the output function in training stage
Y=(H' * OutputWeight)';                         

%% Test using test data without privileged information 
tic
BiasMatrix_test=repmat(Bias,1,NumberofTestingData);
tempH_test=Weight*TV.P + BiasMatrix_test;

% Calculate the activation function
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        H_test = sin(tempH_test);        
    case {'hardlim'}
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        H_test = tribas(tempH_test);        
    case {'radbas'}
        H_test = radbas(tempH_test);               
end
H_test = [TV.P;H_test];

% Calculate the output function in test stage
TY=(H_test' * OutputWeight)';                       
TestingTime=toc;

%% Calculate training and test accuracy
% regression case
if option.Type == 0
    TrainingAccuracy=sqrt(mse(T - Y)); 
    TestingAccuracy=sqrt(mse(TV.T - TY));            

% classification case
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