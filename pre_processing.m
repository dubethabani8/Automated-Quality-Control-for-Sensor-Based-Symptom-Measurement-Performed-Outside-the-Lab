% [dr, drl, ai, g] = pre_processing(filename, lambda)
%
% This function applies pre-processing steps to accelerometer data
% The pre-processing includes interpolating to uniform
% sampling rate (using cubic spline interpolation) and code to remove the effect of
% orientation changes.
%
% Inputs:  filename - specifies the name of the file which contains the data to be
%                     pre-processed.
%          lambda - specifies the parameter for the L1-trend filter
%
% Outputs: %  dr  - stores the amplitude of the dynamic component
%             drl - stores the log10 of the amplitude of the dynamic component
%             ai  - stores the interpolated 3-d accelerometer data
%             g   - stores the estimated acceleration force due to gravity
%
% CC BY-SA 3.0 Attribution-Sharealike 3.0, Y.P. Raykov and M.A. Little. If you use this
% code in your research, please cite:
% R. Badawy, Y.P. Raykov, L.J.W. Evers, B.R. Bloem, M.J. Faber, A. Zhan, K. Claes, M.A. Little (2018)
% "Automated quality control for sensor based symptom measurement performed outside the lab",
% Sensors, (18)4:1215
% This implementation follows the description in that paper.

function [dr, drl, ai, g] = pre_processing(filename, lambda)

Y = csvread(filename);

t = Y(:,1)-Y(1,1);
a = Y(:,2:4);

srate = 120;
    
% Cubic spline interpolate to uniform sampling rate
ti = (0:1/srate:t(end))';
ai = interp1(t,a,ti,'cubic');
t = ti;
a = ai;

% Estimate long timescale gravitational component as continuous piecewise linear change
% with Laplacian i.i.d. perturbations
g(:,1) = utils_filter_l1tf(a(:,1),lambda);
g(:,2) = utils_filter_l1tf(a(:,2),lambda);
g(:,3) = utils_filter_l1tf(a(:,3),lambda);

% Estimate non-gravitational component
d = a-g;
dr = sqrt(sum(d.^2,2));
drl = log10(dr);

drl(isinf(drl)) = 0;

end