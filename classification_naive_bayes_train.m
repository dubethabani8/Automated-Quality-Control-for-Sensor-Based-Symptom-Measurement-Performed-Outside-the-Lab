% params = classification_naive_bayes_train(y, train_labels)
%
% Trains a multinomial naive Bayes binary classifier
%
% Inputs:  y            - KxT matrix storing a K-bin histogram for each observation t
%                         which is computed based on the posterior draws for the indicator variable z_t
%                         associated with point t. Hence each row of y contains the 
%                         posterior frequency of z for each data point t 
%
%          train_labels - Tx1 vector storing the training labels for each point 
%
% Outputs: params - Kx2 matrix storing the estimated training parameters of the multinomial 
%                   naive Bayes classifier
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

function params = classification_naive_bayes_train(y, train_labels)

    T = size(y,2);
    K = size(y,1);
    for k=1:K
        IndClass1 = (train_labels==1);      
        params(k,1) = sum(y(k,IndClass1));
        IndClass2 = (train_labels==2);
        params(k,2) = sum(y(k,IndClass2));
    end
    params = params./(T*sum(y(:,1)));
    params = params + 0.0001; %Add fixed prior probability for each class
end

