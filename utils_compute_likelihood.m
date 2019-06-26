% likelihood = utils_compute_likelihood(data_struct, theta, K)
%
% Computes the conditional probability of each point given its state
% parameters. Note that this conditional probability ignores the effect of the
% transition matrix which is accounted later on in the backwards pass.
%
% Inputs:  data_struct - a structure that contains the raw input data and
%                        reformatted form of the data which speed ups the
%                        updates: data_struct.obs contains the input dxT
%                        matrix; data_struct.X contains the extended
%                        (d*r)xT matrix
%                      
%          theta       -  a structure that contains the state parameters: 
%                         theta.invSigma is dxdxK matrix containing the state specific 
%                         AR process noise; theta.A is 1xrxK matrix
%                         containing the state spacific AR coefficients; 
%                         theta.mu is dxK matrix containing the state
%                         specific AR offset.
%
%          K           - a single value denoting the truncation level for
%                        the number of states in the underlying infinite HMM
%
% Outputs: likelihood - KxT matrix containing the conditional probabilities
%                       of each point for each state
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

function likelihood = utils_compute_likelihood(data_struct,theta,K)
       
        invSigma = theta.invSigma;
        A = theta.A;
        X = data_struct.X;
        
        T = size(data_struct.obs,2);
        dimu = size(data_struct.obs,1);
        
        log_likelihood = zeros(K,T);
        if isfield(theta,'mu')   
            mu = theta.mu;       
            for k=1:K
                    
                cholinvSigma = chol(invSigma(:,:,k));
                dcholinvSigma = diag(cholinvSigma);
                
                u = cholinvSigma*(data_struct.obs - A(:,:,k)*X-mu(:,k*ones(1,T)));
                    
                log_likelihood(k,:) = -0.5*sum(u.^2,1) + sum(log(dcholinvSigma));
            end
        else         
            for k=1:K
                
                cholinvSigma = chol(invSigma(:,:,k));
                dcholinvSigma = diag(cholinvSigma);
                    
                u = cholinvSigma*(data_struct.obs - A(:,:,k)*X);                    
                log_likelihood(k,:) = -0.5*sum(u.^2,1) + sum(log(dcholinvSigma));
                    
            end
            
        end
        normalizer = max(log_likelihood,[],1);        
        log_likelihood = log_likelihood - normalizer(ones(K,1),:);
        likelihood = exp(log_likelihood);
        normalizer = normalizer - (dimu/2)*log(2*pi);
       
end