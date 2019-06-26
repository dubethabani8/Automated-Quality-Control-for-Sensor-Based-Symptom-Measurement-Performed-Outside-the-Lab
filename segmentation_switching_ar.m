% [state_ind_store, likelihood] = segmentation_switching_ar(data_struct, model, K, Niter)
%
% Train nonparametric switching autoregressive process using Gibbs
% sampling.
%
% Inputs:  data_struct   - structure that contains the raw input data and
%                          reformatted form of the data which speed ups the
%                          updates: data_struct.obs contains the input dxT
%                          matrix; data_struct.X contains the extended
%                          (d*r)xT matrixTx1 vector storing the training
%                          labels for each point.
%
%          model         - structure that contains the model
%                          hyperparameters which can be split to prior state parameters 
%                          and prior pramaters defining the underlying infinite HMM. These are stored in fields
%                          component.params and HMMmodel.params from the
%                          structure model. Each of the two fields contains
%                          a structure on its own with mutliple parameter
%                          values, vectors or matrices (explained in detail where used): the component
%                          parameters structure is specified in functions
%                          utils_initialize and sample_theta; the HMM
%                          parameters structure is specified in functions
%                          utils_initialize, sample_tables and
%                          sample_trans_par.
%
%          K             - single value denoting the truncation level for
%                          the number of states in the underlying infinite HMM
%
% Outputs: state_ind_store - structure containing the sampled state
%                            indicator values for each iteration. The structure contains Niter 
%                            number of fields each storing 1xT vector z with the sampled 
%                            state indicator values at that iteration. 
%          likelihood      - 1xNiter vector with the complete data
%                            log-likelihood computed at each iteration.
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

function [state_ind_store, likelihood] = segmentation_switching_ar(Y, model, K, Niter)
    
    % Build initial structures for parameters and sufficient statistics:
    prior_state_params  = model.component.params;  % structure containing the observation model parameters
    [theta(1), update_stats(1), trans_counts(1), prior_HMM_params, data_struct] = switching_ar_utils_initialize(prior_state_params,Y,K);
    prior_HMM_params = model.HMMmodel.params; % structure for the HMM parameters
         
    trans_par(1) = switching_ar_sample_trans_par(trans_counts(1),prior_HMM_params);
    theta(1)     = switching_ar_sample_theta(theta(1),update_stats(1),prior_state_params);

    % Run Sampler
    for n=1:Niter    
        [state_ind, ind_struct, trans_counts, likelihood(n)] = switching_ar_sample_z(data_struct,trans_par,theta); 
        update_stats = switching_ar_utils_update_stats(data_struct, ind_struct, trans_counts);  
        trans_counts = switching_ar_sample_tables(trans_counts,prior_HMM_params,trans_par.beta_vec,K); %%Nonparametric HMMs
        trans_par  = switching_ar_sample_trans_par(trans_counts,prior_HMM_params); %%Parametric HMM
        theta      = switching_ar_sample_theta(theta,update_stats,prior_state_params); % AR specific 
        % Store the sampled parameters at each iteration  
        state_ind_store(n) = state_ind;
    end
end