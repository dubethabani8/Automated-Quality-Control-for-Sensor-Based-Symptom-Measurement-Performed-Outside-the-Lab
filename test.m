% Demo code for quality control on gait data using iHMM-AR with naive Bayes
% classification.
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

addpath('supplementary'); % add suplementary files if you do not have Minka's lightspeed library installed
close all;

%% Pre-processing stage and plots
% Load data files
filenameTrain = '2017-08-18T14-59-52-530Z49.wav';
lambda = 1000;
[dr1, drl1, ai1, g1] = audio_pre_processing(filenameTrain, lambda);

filenameTest = '2017-09-28T14-17-07-280Z18.wav';
lambda = 1000;
[dr2, drl2, ai2, g2] = audio_pre_processing(filenameTest, lambda);

figure; 
subplot(3,1,1);
plot(ai1);
set(gca, 'XTick', [])
title('Training gait data')
axis tight
grid on
subplot(3,1,2);
plot(g1);
title('Gravitational component')
axis tight
grid on
set(gca, 'XTick', [])
subplot(3,1,3);
plot(drl1);
title('Amplitude - dynamic component')
axis tight
grid on
xlabel('Time(s)')
set(gca, 'XTick', [0,600,1200,1800,2400,3000,3600])
set(gca, 'XTickLabel', {'0','5','10','15','20','25','30'})

figure; 
subplot(3,1,1);
plot(ai2);
set(gca, 'XTick', [])
title('Test gait data')
axis tight
grid on
subplot(3,1,2);
plot(g2);
title('Gravitational component')
axis tight
grid on
set(gca, 'XTick', [])
subplot(3,1,3);
plot(drl2);
xlabel('Time(s)')
set(gca, 'XTick', [0,600,1200,1800,2400,3000,3600])
set(gca, 'XTickLabel', {'0','5','10','15','20','25','30'})
title('Amplitude - dynamic component')
axis tight
grid on

Y = [dr1; dr2];
Y = Y'; 
% Downsample by factor 4 in case of gait tests
Y = filter(ones(1,4)/4,1,Y); % anti-aliasing via moving average filter
Y_down = downsample(Y,4);

%% Segmentation stage 

[d,T] = size(Y); % Dimensions and length of the input

r = 30; % maximum order for the state specific AR models

P = inv(diag(0.05*ones(1,d*r))); % inverse covariance along rows of A 
sig0 = 1; % variance for the mean process noise
meanSigma = 10*eye(d); % mean covariance for the covariance of the AR process noise

K = 30;   % truncation level for mode transition distributions
MaxIter = 300;  % number of iterations of the Gibbs sampler

% Set hyperparameters

clear model

model.component.params.r = r; %choose the maximum AR order for each state specific AR
m = d*r;

% Mean and covariance for A matrix:
model.component.params.M  = zeros([d m]);

% Inverse covariance along rows of A (sampled Sigma acts as
% covariance along columns):
model.component.params.P =  P(1:m,1:m);

% Mean and covariance for mean of process noise:
model.component.params.mu0 = zeros(d,1);
model.component.params.cholSigma0 = chol(sig0*eye(d));
        
% Degrees of freedom and scale matrix for covariance of process noise:
model.component.params.nu = d + 2; 
model.component.params.nu_delta = (model.component.params.nu-d-1)*meanSigma;

% Sticky HDP-HMM parameter settings:
model.HMMmodel.params.gamma0 = 10;
model.HMMmodel.params.alpha0 = 5; 
model.HMMmodel.params.kappa0 = 10;

[state_ind_store, likelihood] = segmentation_switching_ar(Y_down, model, K, MaxIter);
[max_likelihood_val, max_likelihood_iter] = max(likelihood);
fprintf('Switching iHMM-AR finished\n');

%% Segmentation result plots
Z = state_ind_store(1).z';
Z = [ones(r,1).*Z(1); Z];
ChangePoints = (diff(Z)~=0);
ChangePointsLoc = find(ChangePoints==1);
ChangePointsLoc(end+1) = length(Y_down);
ChangePointsLoc = [1; ChangePointsLoc]; 
C=[{'m'},{'c'},{'r'},{'y'},{'k'},mat2cell(rand([3,max(Z)-5])',ones(1,max(Z)-5))'];

figure;

subplot(3,1,1);
plot(Y_down(1:end));hold on;
for k = 1:(length(ChangePointsLoc)-1)
    X = rectangle('Position',[ChangePointsLoc(k),-10,(ChangePointsLoc(k+1)-ChangePointsLoc(k)),20],'FaceColor',C{Z(ChangePointsLoc(k)+1)});
    X.FaceColor(4)=.2;
    X.EdgeColor(4)=.2;
end
set(gca, 'XTick', [])
title('Segmentation indicators - 2nd iteration')
axis tight
grid on

subplot(3,1,2)
plot(Y_down(1:end));hold on;
Z = state_ind_store(5).z';  
Z = [ones(r,1).*Z(1); Z];
ChangePoints = (diff(Z)~=0);
ChangePointsLoc = find(ChangePoints==1);
ChangePointsLoc(end+1) = length(Y_down);
ChangePointsLoc = [1; ChangePointsLoc]; 
C=[{'m'},{'c'},{'r'},{'y'},{'k'},mat2cell(rand([3,max(Z)-5])',ones(1,max(Z)-5))'];
for k = 1:(length(ChangePointsLoc)-1)
    X = rectangle('Position',[ChangePointsLoc(k),-10,(ChangePointsLoc(k+1)-ChangePointsLoc(k)),20],'FaceColor',C{Z(ChangePointsLoc(k)+1)});
    X.FaceColor(4)=.2;
    X.EdgeColor(4)=.2;
end
set(gca, 'XTick', [])
title('Segmentation indicators - 5th iteration')
axis tight
grid on

subplot(3,1,3)
plot(Y_down(1:end));hold on;
Z = state_ind_store(5).z'; 
Z = [ones(r,1).*Z(1); Z];
ChangePoints = (diff(Z)~=0);
ChangePointsLoc = find(ChangePoints==1);
ChangePointsLoc(end+1) = length(Y_down);
ChangePointsLoc = [1; ChangePointsLoc]; 
C=[{'m'},{'c'},{'r'},{'y'},{'k'},mat2cell(rand([3,max(Z)-5])',ones(1,max(Z)-5))'];
for k = 1:(length(ChangePointsLoc)-1)
    X = rectangle('Position',[ChangePointsLoc(k),-10,(ChangePointsLoc(k+1)-ChangePointsLoc(k)),20],'FaceColor',C{Z(ChangePointsLoc(k)+1)});
    X.FaceColor(4)=.2;
    X.EdgeColor(4)=.2;
end
xlabel('Time(s)')
set(gca, 'XTick', [0,300,600,900,1200,1500,1800])
set(gca, 'XTickLabel', {'0','10','20','30','40','50','60'})
title('Segmentation indicators - maximum model likelihood iteration')
axis tight
grid on


figure; 
subplot(3,1,1);
plot(ai1);
set(gca, 'XTick', [])
title('Training gait accelerometer data')
axis tight
grid on
subplot(3,1,2);
plot(drl1);
set(gca, 'XTick', [])
title('Amplitude of dynamic component')
axis tight
grid on
subplot(3,1,3);
plot(Y_down(1:round(length(drl1)/4)));hold on;
Z = state_ind_store(max_likelihood_iter).z(1:(round(length(drl1)/4)-r))'; 
Z = [ones(r,1).*Z(1); Z];
ChangePoints = (diff(Z)~=0);
ChangePointsLoc = find(ChangePoints==1);
ChangePointsLoc(end+1) = round(length(drl1)/4);
ChangePointsLoc = [1; ChangePointsLoc]; 
C=[{'m'},{'c'},{'r'},{'y'},{'k'},mat2cell(rand([3,max(Z)-5])',ones(1,max(Z)-5))'];
for k = 1:(length(ChangePointsLoc)-1)
    X = rectangle('Position',[ChangePointsLoc(k),-10,(ChangePointsLoc(k+1)-ChangePointsLoc(k)),20],'FaceColor',C{Z(ChangePointsLoc(k)+1)});
    X.FaceColor(4)=.2;
    X.EdgeColor(4)=.2;
end
xlabel('Time(s)')
set(gca, 'XTick', [0,150,300,450,600,750,900])
set(gca, 'XTickLabel', {'0','5','10','15','20','25','30'})
title('Segmentation indicators')
axis tight
grid on

figure; 
subplot(3,1,1);
plot(ai2);
set(gca, 'XTick', [])
title('Test gait accelerometer data')
axis tight
grid on
subplot(3,1,2);
plot(drl2);
set(gca, 'XTick', [])
title('Amplitude of dynamic component')
axis tight
grid on
subplot(3,1,3);
plot(Y_down((round(length(drl1)/4)+1):end));hold on;
Z = state_ind_store(max_likelihood_iter).z(((round(length(drl1)/4)-r+1):end))'; 
ChangePoints = (diff(Z)~=0);
ChangePointsLoc = find(ChangePoints==1);
ChangePointsLoc(end+1) = length(Y_down((round(length(drl1)/4)+1):end));
ChangePointsLoc = [1; ChangePointsLoc]; 
C=[{'m'},{'c'},{'r'},{'y'},{'k'},mat2cell(rand([3,max(Z)-5])',ones(1,max(Z)-5))'];
for k = 1:(length(ChangePointsLoc)-1)
    X = rectangle('Position',[ChangePointsLoc(k),-10,(ChangePointsLoc(k+1)-ChangePointsLoc(k)),20],'FaceColor',C{Z(ChangePointsLoc(k)+1)});
    X.FaceColor(4)=.2; 
    X.EdgeColor(4)=.2;
end
xlabel('Time(s)')
set(gca, 'XTick', [0,150,300,450,600,750,900])
set(gca, 'XTickLabel', {'0','5','10','15','20','25','30'})
title('Segmentation indicators')
axis tight
grid on

%% Classification stage

train_labels = load('train_labels_audio.csv');
train_labels = train_labels(r+1:end); % we cannot estimate the state indicators for the first r points, hence in this demo we discard them

state_ind_matrix = [];
for d = 1:MaxIter
    state_ind_matrix = [state_ind_matrix; state_ind_store(d).z];
end
train_ind_hist = hist(state_ind_matrix(:, 1:(round(length(drl1)/4)-r)),[1:K],1); % Create a histogram of the 
                                                                                 % posterior indicators for the training test
                                                                                 
test_labels = load('test_labels_audio.csv');
test_ind_hist = hist(state_ind_matrix(:,(round(length(drl1)/4)-r+1):end),[1:K],1); % Create a 
                                                                                   % histogram of the posterior indicators for testing

prob_train = classification_naive_bayes_train( train_ind_hist, train_labels);

% Remove state indicators pointing to states with have very low or no support
s = 1;
for k=1:length(prob_train)
    if sum(prob_train(k,:))>0.001 
        prob_train_real(s,:) = prob_train(k,:);
        test_ind_hist_real(s,:) = test_ind_hist(k,:);
        s=s+1;
    end
end

%% Classification result plots
figure;
subplot(3,1,1);
plot(ai1);
set(gca, 'XTick', [])
% xlabel('Time(s)')
% set(gca, 'XTick', [0,600,1200,1800,2400,3000,3600])
% set(gca, 'XTickLabel', {'0','5','10','15','20','25','30'})
title('Labeled training gait data')
axis tight
grid on
subplot(3,1,2);
plot(Y_down(1:round(length(drl1)/4)));hold on;
Z = state_ind_store(max_likelihood_iter).z(1:(round(length(drl1)/4)-r))';  
Z = [ones(r,1).*Z(1); Z];
ChangePoints = (diff(Z)~=0);
ChangePointsLoc = find(ChangePoints==1);
ChangePointsLoc(end+1) = round(length(drl1)/4);
ChangePointsLoc = [1; ChangePointsLoc]; 
C=[{'m'},{'c'},{'r'},{'y'},{'k'},mat2cell(rand([3,max(Z)-5])',ones(1,max(Z)-5))'];
for k = 1:(length(ChangePointsLoc)-1)
    X = rectangle('Position',[ChangePointsLoc(k),-10,(ChangePointsLoc(k+1)-ChangePointsLoc(k)),20],'FaceColor',C{Z(ChangePointsLoc(k)+1)});
    X.FaceColor(4)=.2; 
    X.EdgeColor(4)=.2;
end
set(gca, 'XTick', [])
% xlabel('Time(s)')
% set(gca, 'XTick', [0,150,300,450,600,750,900])
% set(gca, 'XTickLabel', {'0','5','10','15','20','25','30'})
title('Segmentation indicators')
axis tight
grid on

subplot(3,1,3);
plot(Y_down(1:round(length(drl1)/4)));hold on;
Z = train_labels;
Z = [ones(r,1).*Z(1); Z];
ChangePoints = (diff(Z)~=0);
ChangePointsLoc = find(ChangePoints==1);
ChangePointsLoc(end+1) = length(train_labels)+r;
ChangePointsLoc = [1; ChangePointsLoc]; 
C=[{'r'},{'k'}];
for k = 1:(length(ChangePointsLoc)-1)
    X = rectangle('Position',[ChangePointsLoc(k),-10,(ChangePointsLoc(k+1)-ChangePointsLoc(k)),20],'FaceColor',C{Z(ChangePointsLoc(k)+1)});
    X.FaceColor(4)=.2;
    X.EdgeColor(4)=.2;
end
xlabel('Time(s)')
set(gca, 'XTick', [0,150,300,450,600,750,900])
set(gca, 'XTickLabel', {'0','5','10','15','20','25','30'})
title('Train labels')
axis tight
grid on

[predict_class, class_prob] = classification_naive_bayes_test(test_ind_hist_real, prob_train_real);

figure;

subplot(4,1,1);
plot(ai2);
set(gca, 'XTick', [])
title('Labeled test gait data')
axis tight
grid on

subplot(4,1,2);
plot(Y_down((round(length(drl1)/4)+1):end));hold on;
Z = state_ind_store(max_likelihood_iter).z(((round(length(drl1)/4)-r+1):end))';
Z = [ones(r,1).*Z(1); Z];
ChangePoints = (diff(Z)~=0);
ChangePointsLoc = find(ChangePoints==1);
ChangePointsLoc(end+1) = length(Y_down((round(length(drl1)/4)+1):end));
ChangePointsLoc = [1; ChangePointsLoc]; 
C=[{'m'},{'c'},{'r'},{'y'},{'k'},mat2cell(rand([3,max(Z)-5])',ones(1,max(Z)-5))'];
for k = 1:(length(ChangePointsLoc)-1)
    X = rectangle('Position',[ChangePointsLoc(k),-10,(ChangePointsLoc(k+1)-ChangePointsLoc(k)),20],'FaceColor',C{Z(ChangePointsLoc(k)+1)});
    X.FaceColor(4)=.2; 
    X.EdgeColor(4)=.2;
end
set(gca, 'XTick', [])
title('Segmentation indicators')
axis tight
grid on

subplot(4,1,3);
plot(Y_down((round(length(drl1)/4)+1):end));hold on;
Z = test_labels;
ChangePoints = (diff(Z)~=0);
ChangePointsLoc = find(ChangePoints==1);
ChangePointsLoc(end+1) = length(test_labels);
ChangePointsLoc = [1; ChangePointsLoc]; 
C=[{'r'},{'k'}];
for k = 1:(length(ChangePointsLoc)-1)
    X = rectangle('Position',[ChangePointsLoc(k),-10,(ChangePointsLoc(k+1)-ChangePointsLoc(k)),20],'FaceColor',C{Z(ChangePointsLoc(k)+1)});
    X.FaceColor(4)=.2; 
    X.EdgeColor(4)=.2;
end
set(gca, 'XTick', [])
title('Test labels')
axis tight
grid on

subplot(4,1,4);
plot(Y_down((round(length(drl1)/4)+1):end));hold on;
Z = predict_class;
ChangePoints = (diff(Z)~=0);
ChangePointsLoc = find(ChangePoints==1);
ChangePointsLoc(end+1) = length(test_labels);
ChangePointsLoc = [1; ChangePointsLoc]; 
C=[{'r'},{'k'}];
for k = 1:(length(ChangePointsLoc)-1)
    X = rectangle('Position',[ChangePointsLoc(k),-10,(ChangePointsLoc(k+1)-ChangePointsLoc(k)),20],'FaceColor',C{Z(ChangePointsLoc(k)+1)});
    X.FaceColor(4)=.2; 
    X.EdgeColor(4)=.2;
end
xlabel('Time(s)')
set(gca, 'XTick', [0,150,300,450,600,750,900])
set(gca, 'XTickLabel', {'0','5','10','15','20','25','30'})
title('Estimated quality control indicators')
axis tight
grid on
