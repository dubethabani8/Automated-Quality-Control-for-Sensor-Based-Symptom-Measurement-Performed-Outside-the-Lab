% theta = switching_ar_sample_theta(theta,update_stats,obsModel)
%
% Samples the state parameters A, mu and invSigma which specify the state
% specific autoregressive processes. A containes the AR coefficients, mu is
% the AR offset parameter and invSigma is the inverse of the AR process
% noise
%
% Inputs: theta        -  a structure that contains the state parameters: 
%                         theta.invSigma is dxdxK matrix containing the state specific 
%                         AR process noise; theta.A is 1xrxK matrix
%                         containing the state spacific AR coefficients; 
%                         theta.mu is dxK matrix containing the state
%                         specific AR offset.
%
%         update_stats - a structure containing sufficient statistics
%                        required for the efficient parameter updates.
%                        Using X to denote the reformatted form of the
%                        dataand Y the input raw form:
%                        -> update_stats.card is 1xK vector obtained by a row sum of the transition counts  
%                        -> update_stats.card is rxrxK matrix obtained by the sum of products X*X'  
%                        -> update_stats.card is 1xrxK matrix obtained by the sum of products Y*X' 
%                        -> update_stats.card is 1x1xK vector obtained by the sum of products Y*Y'  
%                        -> update_stats.card is 1xK matrix obtained by the sum over Y  
%                        -> update_stats.card is rxK matrix obtained by the sum over X 
%
%         prior_params - structure containing various model prior parameters.
%                        In this function we use the fields r, M, P, mu0,cholSigma0, nu, nu_delta 
%                        -> M          - mean for the AR coefficients A (set to 0 for the demo)
%                        -> P          - inverse covariance along the rows
%                                        of A (used to sample the covariance between the coefficients)                         
%                        -> mu0        - prior mean for the mean of the AR process noise (set to 0 for the demo)
%                        -> cholSigma0 - prior covariance for the mean of the AR process noise
%                        -> nu         - degress of freedom for the covariance of the AR process noise
%                        -> nu_delta   - scale matrix for the covariance
%                                        of the AR process noise 
%
% Outputs: theta - a structure that contains the update state parameters
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

function theta = switching_ar_sample_theta(theta, update_stats, prior_params)

    nu = prior_params.nu;
    nu_delta = prior_params.nu_delta;
    store_card = update_stats.card;

    if size(store_card,1)==1
        store_card = store_card';
    end
    K = size(store_card);
        
    invSigma = theta.invSigma;
    A = theta.A;
    mu = theta.mu;
        
    store_XX = update_stats.XX;
    store_YX = update_stats.YX;
    store_YY = update_stats.YY;
    store_sumY = update_stats.sumY;
    store_sumX = update_stats.sumX;
        
    P = prior_params.P;
    M = prior_params.M;
    MP = prior_params.M*prior_params.P;
    MKP = MP*prior_params.M';
        
    numIter = 10;    
    mu0 = prior_params.mu0;
    cholSigma0 = prior_params.cholSigma0;
    Lambda0 = inv(prior_params.cholSigma0'*prior_params.cholSigma0);
    theta0 = Lambda0*prior_params.mu0;
        
    dimu = size(nu_delta,1);   
    for k=1:K

           if store_card(k,1)>0 
              for n=1:numIter
                        
                  %% Given X, Y get sufficient statistics
                  Sxx       = store_XX(:,:,k) + P;
                  Syx       = store_YX(:,:,k) + MP - mu(:,k)*store_sumX(:,k,1)';
                  Syy       = store_YY(:,:,k) + MKP ...
                            - mu(:,k)*store_sumY(:,k,1)' - store_sumY(:,k,1)*mu(:,k)' + store_card(k)*mu(:,k)*mu(:,k)';
                  SyxSxxInv = Syx/Sxx;
                  Sygx      = Syy - SyxSxxInv*Syx';
                  Sygx      = (Sygx + Sygx')/2;
                        
                  % Sample Sigma given s.stats
                  [sqrtSigma sqrtinvSigma] = randgen_invwishart(Sygx + nu_delta,nu+store_card(k,1));

                  invSigma(:,:,k)      = sqrtinvSigma'*sqrtinvSigma;
                        
                   % Sample A given Sigma and s.stats

                  cholinvSxx = chol(inv(Sxx));
                  A(:,:,k) = randgen_matrixNormal(SyxSxxInv,sqrtSigma,cholinvSxx);
                        
                  % Sample mu given A and Sigma
                  Sigma_n = inv(Lambda0 + store_card(k,1)*invSigma(:,:,k));
                  mu_n = Sigma_n*(theta0 + invSigma(:,:,k)*(store_sumY(:,k,1)-A(:,:,k)*store_sumX(:,k,1)));
                        
                  mu(:,k) = mu_n + chol(Sigma_n)'*randn(dimu,1);
                        
              end
           else
                    
             [sqrtSigma, sqrtinvSigma] = randgen_invwishart(nu_delta,nu);
             invSigma(:,:,k)      = sqrtinvSigma'*sqrtinvSigma;                 
             cholinvK  = chol(inv(P));
             A(:,:,k) = randgen_matrixNormal(M,sqrtSigma,cholinvK);        
             mu(:,k)  = mu0 + cholSigma0'*randn(dimu,1);                    
           end
                
    end
        
    theta.invSigma = invSigma;
    theta.A = A;
    theta.mu =  mu;

end

function [sqrtx,sqrtinvx] = randgen_invwishart(sigma,df,di)
%  Generate inverse Wishart random matrix
%  [sqrtx,sqrtinvx,di]=randgen_invwishart(sigma,df) generates a random matrix W=sqrtinvx'*sqrtinvx
%  from the inverse Wishart distribution with parameters sigma and df.  The inverse of W
%  has the Wishart distribution with covariance matrix inv(sigma) and df
%  degrees of freedom.
%

    n = size(sigma,1);
    if (df<n) % require this to ensure invertibility
    error('randiwish:BadDf',...
         'Degrees of freedom must be no smaller than the dimension of SIGMA.');
    end

    % Get Cholesky factor for inv(sigma) unless that's already done
    if nargin<3
        d = chol(sigma);
        di = d'\eye(size(d));  % either take inverse here and scale chol of
        %randwishart sample and then take inverse of sample, or take inverse of
        %sample and then scale after w/o the inverse.
    end

    a = randwishart(df/2,n);
    sqrtinvx = sqrt(2)*a*di;
    sqrtx = (sqrtinvx\eye(size(sqrtinvx)))';
end
    

function A = randgen_matrixNormal(M,sqrtV,sqrtinvK)
% Draw samples from Matrix Normal distribution
    mu = M(:);
    sqrtsigma = kron(sqrtinvK,sqrtV);

    A = mu + sqrtsigma'*randn(length(mu),1);
    A = reshape(A,size(M));
    
end
