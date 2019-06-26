% update_stats = switching_ar_utils_update_stats(data_struct, ind_struct, trans_counts)
%
% Update the sufficient statistics for each state to speed up the sampling
% for the state parametes theta.
%
% Input:  data_struct - a structure that contains the raw input data and
%                       reformatted form of the data which speed ups the
%                       updates: data_struct.obs contains the input dxT
%                       matrix; data_struct.X contains the extended
%                       (d*r)xT matrix
%        ind_struct   - structure with K fields obsIndzs which all have
%                       sub-fields obsIndzs(k,1).tot  and obsIndzs(k,1).inds: 
%                       obsIndzs(k,1).tot is a scalar denoting the total
%                       number of points belonging to state k; obsIndzs(k,1).inds
%                       is 2xT sparse list which stores the indices of points 
%                       assigned to state k. For example, if only points t and t+1 are 
%                       associated with state k ind_struct.obsIndzs(k,1).inds = {(1,1), t};{(1,2), t+1}
%                       where the first column of the list just records
%                       the order of which points associated with that
%                       state occur.
%        trans_counts - a structure containing the transition counts and
%                       a count of the different transitions pointing to
%                       each state: trans_counts.N is (K+1)xK matrix
%                       counting the number of transitions that have
%                       occured from state i to state j, the extra row occounts 
%                       for the support of creating a new transition; 
%
% Output: update_stats - a structure containing sufficient statistics
%                        required for the efficient parameter updates.
%                        Using X to denote the reformatted form of the
%                        dataand Y the input raw form:
%                      -> update_stats.card is 1xK vector obtained by a row sum of the transition counts  
%                      -> update_stats.card is rxrxK matrix obtained by the sum of products X*X'  
%                      -> update_stats.card is 1xrxK matrix obtained by the sum of products Y*X' 
%                      -> update_stats.card is 1x1xK vector obtained by the sum of products Y*Y'  
%                      -> update_stats.card is 1xK matrix obtained by the sum over Y  
%                      -> update_stats.card is rxK matrix obtained by the sum over X 
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

function update_stats = switching_ar_utils_update_stats(data_struct, ind_struct, trans_counts)

    N = trans_counts.N;
    K = size(N,2);
    unique_z = find(sum(N,1));

    dimu = size(data_struct(1).obs,1);
    dimX = size(data_struct(1).X,1);

    store_XX = zeros(dimX,dimX,K);
    store_YX = zeros(dimu,dimX,K);
    store_YY = zeros(dimu,dimu,K);
    store_sumY = zeros(dimu,K);
    store_sumX = zeros(dimX,K);

    u = data_struct.obs;
    X = data_struct.X;
    
    for k=unique_z
        obsInd = ind_struct.obsIndzs(k,1).inds(1:ind_struct.obsIndzs(k,1).tot);
        store_XX(:,:,k,1) = store_XX(:,:,k,1) + X(:,obsInd)*X(:,obsInd)';
        store_YX(:,:,k,1) = store_YX(:,:,k,1) + u(:,obsInd)*X(:,obsInd)';
        store_YY(:,:,k,1) = store_YY(:,:,k,1) + u(:,obsInd)*u(:,obsInd)';
        store_sumY(:,k,1) = store_sumY(:,k,1) + sum(u(:,obsInd),2);
        store_sumX(:,k,1) = store_sumX(:,k,1) + sum(X(:,obsInd),2);
    end
                     
    update_stats.card = sum(trans_counts.N,1);
    update_stats.XX = store_XX;
    update_stats.YX = store_YX;
    update_stats.YY = store_YY;
    update_stats.sumY = store_sumY;
    update_stats.sumX = store_sumX;
        
end