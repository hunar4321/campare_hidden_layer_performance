%% The following is a complete example of single layer neural network-
%% from scratch using Matlab (no frameworks - no blackbox functions!)
%% @author: Hunar Ahmad Abdulrahman

%% xor data

n=1000;
X=zeros(8, n);
y=zeros(1, n);
for i=1:n;
    data=[1,0];
    dat1 = randsample(data,1);
    dat2 = randsample(data,1);
    dat3 = randsample(data,1);
    dat4 = randsample(data,1);    
    dat5 = randsample(data,1);
    dat6 = randsample(data,1);
    dat7 = randsample(data,1);
    dat8 = randsample(data,1);   
    
    out = double_double_gen_xor(dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8);
    
    X(1,i)=dat1; X(2,i)=dat2; X(3, i)=dat3; X(4,i)=dat4;
    X(5,i)=dat5; X(6,i)=dat6; X(7, i)=dat7; X(8,i)=dat8;
    
    
    y(1,i)=out;
end
X = X'; y=y';

% X=[0,0; 1,1; 1,0; 0,1];  %inputs
% y=[0; 0; 1; 1];          %outputs

% model settings 
learn_rate=0.0001;
in=8; nodes=12; out =1;
% weight initialization
W1=randn(in,nodes);
W2=randn(nodes,out);
%% training loop...
for i=1:5000;
    % feedforward
    z1=X*W1;
    X2=sin(z1); % "sin" used as "activation function" for simplicity
    z2=X2*W2;
    yhat=sin(z2);
    % backpropagation
    delta2 = (y-yhat).*cos(z2); % "cos" is derivative of "sin"
    W2=W2+(X2'*delta2) *learn_rate;
    delta1 = delta2*W2'.*cos(z1);
    W1=W1+(X'*delta1) *learn_rate;
    % optional line ( squared error for visualization);
    mse(i)=mean((y-yhat).^2);
end
%% output & error visualization
plot(mse); title('MSE');
[yhat, y]; % compare the predictions with the true labels

total_params=in*nodes
mse(end)
