% [state_ind, INDS, state_counts, likelihood] = switching_ar_sample_z(data_struct, dist_struct, theta)
%
% Samples the state indicator variables z stored in state_ind.z which
% denote which state is associated with each observation.
% 
% Inputs:  data_struct - structure that contains the raw input data and
%                        reformatted form of the data which speed ups the
%                        updates: data_struct.obs contains the input dxT
%                        matrix; data_struct.X contains the extended
%                        (d*r)xT matrix; 
%
%          trans_par   - structure containing the sampled transition
%                        parameters: pi_z contains the (K+1)xK transition
%                        matrix; pi_init contains the 1xK initial transition
%                        weights; beta_vec is 1xK vector containing the top level mixing
%                        paramters
%                      
%          theta       - a structure that contains the state parameters: 
%                        theta.invSigma is dxdxK matrix containing the state specific 
%                        AR process noise; theta.A is 1xrxK matrix
%                        containing the state spacific AR coefficients; 
%                        theta.mu is dxK matrix containing the state
%                        specific AR offset.
%
% Outputs: state_ind     - structure with a single field z which contains 1xT
%                          vector with the associate state idicator values for each point 
%          ind_struct    - structure with K fields obsIndzs which all have
%                          sub-fields obsIndzs(k,1).tot  and obsIndzs(k,1).inds: 
%                          obsIndzs(k,1).tot is a scalar denoting the total
%                          number of points belonging to state k; obsIndzs(k,1).inds
%                          is 2xT sparse list which stores the indices of points 
%                          assigned to state k. For example, if only points t and t+1 are 
%                          associated with state k ind_struct.obsIndzs(k,1).inds = {(1,1), t};{(1,2), t+1}
%                          where the first column of the list just records
%                          the order of which points associated with that
%                          state occur.
%                          
%          trans_counts  - a structure containing the transition counts and
%                          a count of the different transitions pointing to
%                          each state: trans_counts.N is (K+1)xK matrix
%                          counting the number of transitions that have
%                          occured from state i to state j, the extra row occounts 
%                          for the support of creating a new transition;    
%
%          likelihood    - a single value reflecting the log likelihood of
%                          the data given the sampled state indicators (and the remaining unknown qunatities) 
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

function [state_ind, ind_struct, trans_counts, likelihood] = switching_ar_sample_z(data_struct, trans_par, theta)

    % Define parameters:
    pi_z = trans_par.pi_z;  % transition distributions with pi_z(i,j) the probability of going from i->j
    pi_init = trans_par.pi_init;  % initial distribution on z_1

    K = size(pi_z,2);  % truncation level for transition distributions
    % Initialize state count matrices:
    N = zeros(K+1,K);

    % Preallocate ind_struct
    T = size(data_struct.obs,2); % the number of data points minus the maximum AR order 
    ind_struct.obsIndzs(1:K,1) = struct('inds',sparse(1,T),'tot',0);
    % Initialize state sequence structure:
    state_ind = struct('z',zeros(1,T));
    
    % Initialize state and sub-state sequences:
    z = zeros(1,T);
    
    % Compute likelihood of each data point for each state
    likelihood = utils_compute_likelihood(data_struct,theta,K);
    
    % Compute backwards messages:
    [bwds_msg, partial_marg] = backwards_message_vec(likelihood, T, pi_z);

    %Pre-allocate
    totSeq = zeros(K,1);
    indSeq = zeros(T,K,1);
    likelihood_contribution = zeros(T,1);
    for t=1:T
        % Sample z(t):
        if (t == 1)
            Pz = pi_init' .* partial_marg(:,1); %probability of z
        else
            Pz = pi_z(z(t-1),:)' .* partial_marg(:,t);
        end
        Pz   = cumsum(Pz);
        z(t) = 1 + sum(Pz(end)*rand(1) > Pz);
        likelihood_contribution(t) = Pz(z(t));
        % Add state to counts matrix:
        if (t > 1)
            N(z(t-1),z(t)) = N(z(t-1),z(t)) + 1;
        else
            N(K+1,z(t)) = N(K+1,z(t)) + 1;  
        end

        totSeq(z(t),1) = totSeq(z(t),1) + 1;
        indSeq(totSeq(z(t),1),z(t),1) = t;

    end
    likelihood = sum(likelihood_contribution);
    state_ind.z = z;

    for j = 1:K
        ind_struct.obsIndzs(j,1).tot  = totSeq(j,1);
        ind_struct.obsIndzs(j,1).inds = sparse(indSeq(:,j,1)');
    end
    trans_counts.N = N; 

end

function [bwds_msg, partial_marg] = backwards_message_vec(likelihood,T,pi_z)

    % Allocate storage space
    K = size(pi_z,2);
    bwds_msg     = ones(K,T);
    partial_marg = zeros(K,T);
    % Compute messages backwards in time
    for tt = T-1:-1:1
    % Multiply likelihood by incoming message:
        partial_marg(:,tt+1) = likelihood(:,tt+1) .* bwds_msg(:,tt+1); 
    % Integrate out z_t:
        bwds_msg(:,tt) = pi_z * partial_marg(:,tt+1);
        bwds_msg(:,tt) = bwds_msg(:,tt) / sum(bwds_msg(:,tt));
    end
    % Compute marginal for first time point
    partial_marg(:,1) = likelihood(:,1) .* bwds_msg(:,1);
end