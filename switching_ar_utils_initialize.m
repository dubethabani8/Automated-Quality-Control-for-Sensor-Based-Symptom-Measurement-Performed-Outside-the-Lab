% [theta, update_stats, state_counts, hyperparams, data_struct, model] = switching_ar_utils_initialize(model, data_struct, K)
%
% Initialize structures storing the model parameters and efficient
% representation of the data (extended lag form) which leads to faster updates in MATLAB
% implementation.
%
% Inputs:  prior_params  - structure containing various model prior parameters.
%                          In this function we use the fields r, M, P, mu0,cholSigma0, nu, nu_delta 
%                          -> r          - maximum order of the the state specific AR models;
%                          -> M          - mean for the AR coefficients A (set to 0 for the demo)
%
%          data_struct   - structure that contains the raw input data and
%                          reformatted form of the data which speed ups the
%                          updates: data_struct.obs contains the input dxT
%                          matrix; data_struct.X contains the extended
%                          (d*r)xT matrixTx1 vector storing the training
%                          labels for each point.
%
%          K             - single value denoting the truncation level for
%                          the number of states in the underlying infinite HMM
%
% Outputs: theta        -  a structure that contains the state parameters: 
%                         theta.invSigma is dxdxK matrix containing the state specific 
%                         AR process noise; theta.A is 1xrxK matrix
%                         containing the state spacific AR coefficients; 
%                         theta.mu is dxK matrix containing the state
%                         specific AR offset.
%
%          update_stats - update_stats - a structure containing sufficient statistics
%                         required for the efficient parameter updates.
%                         Using X to denote the reformatted form of the
%                         dataand Y the input raw form:
%                         -> update_stats.card is 1xK vector obtained by a row sum of the transition counts  
%                         -> update_stats.card is rxrxK matrix obtained by the sum of products X*X'  
%                         -> update_stats.card is 1xrxK matrix obtained by the sum of products Y*X' 
%                         -> update_stats.card is 1x1xK vector obtained by the sum of products Y*Y'  
%                         -> update_stats.card is 1xK matrix obtained by the sum over Y  
%                         -> update_stats.card is rxK matrix obtained by the sum over X 
% 
%          trans_counts -  structure containing the transition counts and
%                          a count of the different transitions pointing to
%                          each state: trans_counts.N is (K+1)xK matrix
%                          counting the number of transitions that have
%                          occured from state i to state j, the extra row occounts 
%                          for the support of creating a new transition;
%                          trans_counts.barM is (K+1)xK matrix counting the
%                          number of different states pointing to each
%                          state, i.e. trans.counts.barM(i,j) counts the number of
%                          different transitions. 
%
%          hyperparams  -  structure to store the transition related (parameters 
%                          characerizing the underlying sticky infinite HMM) hyperparameters 
%
%          data_struct  -  modified version of the input structure
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.


function [theta, update_stats, trans_counts, hyperparams, data_struct] = switching_ar_utils_initialize(prior_params,Y,K)

    dimu = size(prior_params.M,1);
    dimX = size(prior_params.M,2);
                
    theta = struct('invSigma',zeros(dimu,dimu,K),'A',zeros(dimu,dimX,K),'mu',zeros(dimu,K));
                
    update_stats = struct('card',zeros(K,1),'XX',zeros(dimX,dimX,K),'YX',zeros(dimu,dimX,K),'YY',zeros(dimu,dimu,K),'sumY',zeros(dimu,K),'sumX',zeros(dimX,K));
            
    [X,valid] = makeDesignMatrix(Y,prior_params.r);
    data_struct.obs = Y(:,find(valid));
    data_struct.X = X(:,find(valid));

    trans_counts.N = zeros(K+1,K);
    trans_counts.M = zeros(K+1,K);
    trans_counts.barM = zeros(K+1,K);
    trans_counts.sum_w = zeros(1,K);

    hyperparams.alpha0 = 0;
    hyperparams.kappa0 = 0; 

    hyperparams.gamma0 = 0;
    hyperparams.sigma0 = 0;
end

function [X,valid]= makeDesignMatrix(Y,order)

    d = size(Y,1);
    T = size(Y,2);

    X = zeros(order*d,T);

    for lag=1:order
        ii   = d*(lag-1)+1;
        indx = ii:(ii+d-1);
        X(indx, :) = [zeros(d,min(lag,T)) Y(:,1:(T-min(lag,T)))]; 
    end

    if nargout > 1
        valid = ones(1,T);
        valid(1:order) = 0;
    end
end
    