% trans_counts = switching_ar_sample_tables(trans_counts,hyperparams,beta_vec,K)
%
% Sample the number of tables in restaurant i serving dish j given the
% state sequence z_1,...,z_T and the process hyperparameters.
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
%                          HMM) hyperparameters: alpha0 and kappa0
%                          %%%rho0 and alpha0_p_kappa0
%
%          beta_vec      - 1xK vector containing the top level mixing
%                          paramters
%
%          K             - single value denoting the truncation level for
%                          the number of states in the underlying infinite HMM
% 
% Outputs: trans_counts - a structure containing the updated transition counts 
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

function trans_counts = switching_ar_sample_tables(trans_counts,hyperparams,beta_vec,K)
    % Split \alpha and \kappa using \rho:
    %rho0 = hyperparams.rho0
    alpha0 = hyperparams.alpha0; %alpha0 = hyperparams.alpha0_plus_kappa0*(1-rho0)
    kappa0 = hyperparams.kappa0; %kappa0 = hyperparams.alpha0_plus_kappa0*rho0;

    N = trans_counts.N;

    % Sample M, where M(i,j) = # of tables in restaurant i served dish j:
    M = randgen_numtable([alpha0*beta_vec(ones(1,K),:)+kappa0*eye(K); alpha0*beta_vec],N);
    % Sample barM (the table counts for the underlying restaurant), where
    % barM(i,j) = # tables in restaurant i that considered dish j:
    [barM sum_w] = sample_barM(M,beta_vec,alpha0,kappa0);

    trans_counts.M = M;
    trans_counts.barM = barM;
    trans_counts.sum_w = sum_w;
end

function numtable = randgen_numtable(alpha,numdata)
% Samples the number of occupied tables in a Chinese restaurant process
% given concentration parameter alpha (scalar) and number of customers
% numdata (scalar).

    numtable=zeros(size(numdata));
    for i=1:prod(size(numdata))
        numtable(i)=1+sum(rand(1,numdata(i)-1)<ones(1,numdata(i)-1)*alpha(i)./(alpha(i)+(1:(numdata(i)-1)))); 
    end
    numtable(numdata==0)=0;
end

function [barM sum_w] = sample_barM(M,beta_vec,alpha0,kappa0)
% Samples a count variable barM reflecting how often transitions to each state 
% occur as new transitions. barM is also know as the count characterzing the 'oracle' DP
% in infinite HMMs. See direct assignment representation of hierarchical Dirichlet processes f
% or more thorough explanation of the distribution of barM.    
    barM = M;
    sum_w = zeros(size(M,2),1);
    for j=1:size(M,2)
        if kappa0>0 && alpha0>0
            p = kappa0/(alpha0*beta_vec(j) + kappa0); % p = rho0/(beta_vec(j)*(1-rho0) + rho0)
        else
            p = 0;
        end
        sum_w(j) = randbinom(p,M(j,j));
        barM(j,j) = M(j,j) - sum_w(j);
    end

end