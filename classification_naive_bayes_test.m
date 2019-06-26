% [predict_class, class_prob] = classification_naive_bayes_test(y, params)
% Predicts class assignments for input data given the parameters of a pre-trained naive Bayes binary classifier  
%
% Inputs:  y      - KxT matrix storing a K-bin histogram for each observation t
%                   which is computed based on the posterior draws for the indicator variable z_t
%                   associated with point t. Hence each row of y contains the 
%                   posterior frequency of z for each data point t 
%
%          params - Kx2 matrix storing the training parameters of the multinomial 
%                   naive Bayes classifier 
%
% Outputs: predict_class - Tx1 vector containing the estimated class labels
%          class_prob    - Tx2 matrix containing the estimated probabilities 
%          for class 1 and class 2 for each point 
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

function [predict_class, class_prob] = classification_naive_bayes_test(y, params)

    [K,T] = size(y);
    predict_class = ones(T,1);
    class_prob = zeros(T,2);
    for t=1:T
        if sum(y(:,t).*log(params(:,1)))<sum(y(:,t).*log(params(:,2)))
            predict_class(t) = 2;
        end;
        class_prob(t,1) = sum(y(:,t).*log(params(:,1)))/(sum(y(:,t).*log(params(:,1))) + sum(y(:,t).*log(params(:,2))));
        class_prob(t,2) = sum(y(:,t).*log(params(:,2)))/(sum(y(:,t).*log(params(:,1))) + sum(y(:,t).*log(params(:,2))));
    end
    
end