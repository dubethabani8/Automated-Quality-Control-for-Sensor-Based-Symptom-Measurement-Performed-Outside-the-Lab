% trans_par = switching_ar_sample_trans_par(state_counts, hyperparams)
%
% Samples transition matrix and top level DP mixing weights for the underlying infinite 
% HMM.
%
% Inputs:  trans_counts  - a structure containing the transition counts and
%                          a count of the different transitions pointing to
%                          each state: trans_counts.N is (K+1)xK matrix
%                          counting the number of transitions that have
%                          occured from state i to state j, the extra row occounts 
%                          for the support of creating a new transition;
%                          trans_counts.barM is (K+1)xK matrix counting the
%                          number of different states pointing to each
%                          state, i.e. trans.counts.barM(i,j) counts the number of
%                          different transitions
% 
%          hyperparams  -  structure to store the transition related (parameters 
%                          characerizing the underlying sticky infinite
%                          HMM) hyperparameters: alpha0 and kappa0 (rho0 and alpha_p_kappa0)
%
% Outputs: trans_par - structure containing the sampled transition
%                      parameters: pi_z contains the (K+1)xK transition
%                      matrix; pi_init contains the 1xK initial transition
%                      weights; beta_vec is 1xK vector containing the top level mixing
%                      paramters
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

function trans_par = switching_ar_sample_trans_par(trans_counts, hyperparams)

    K = size(trans_counts.N,2); % truncation level for transition distributions

    % Define alpha0 and kappa0:
    alpha0 = hyperparams.alpha0; %%alpha0 = hyperparams.alpha0_plus_kappa0*(1-hyperparams.rho0)
    kappa0 = hyperparams.alpha0; %% kappa0 = hyperparams.alpha0_plus_kappa0*hyperparams.rho0

    N = trans_counts.N;  % N(i,j) = # z_t = i to z_{t+1}=j transitions. N(Kz+1,i) = 1 for i=z_1.
    barM = trans_counts.barM;  % barM(i,j) = # tables in restaurant i that considered dish j
    gamma0 = hyperparams.gamma0;
    beta_vec = randgen_dirichlet([sum(barM,1) + gamma0/K]')';

    pi_z = zeros(K,K);
    for j=1:K
        kappa_vec = zeros(1,K);
        kappa_vec(j) = kappa0;
        pi_z(j,:) = randgen_dirichlet([alpha0*beta_vec + kappa_vec + N(j,:)]')';
    end
    pi_init = randgen_dirichlet([alpha0*beta_vec + N(K+1,:)]')';
    trans_par.pi_z = pi_z;
    trans_par.pi_init = pi_init;
    trans_par.beta_vec = beta_vec;
end

function x = randgen_dirichlet(a)
% randgen_dirichlet  Sample from Dirichlet distribution
%
% X = randgen_dirichlet(A) returns a matrix, the same size as A, where X(:,j)
% is sampled from a Dirichlet(A(:,j)) distribution.

    x = randgamma(a);
    Z = sum(x,1);
    x = x./Z(ones(size(a,1),1),:);
end
