function x = randgamma(a)
% Sample from Gamma distribution
% 
% x = randgamma(A) returns a matrix, the same size as A, where X(i,j)
% is a sample from a Gamma(A(i,j)) distribution.
%
% Gamma(a) has density function p(x) = x^(a-1)*exp(-x)/gamma(a).
    [p,q] = size(a);
    x = gamrnd(a,(ones(p,q)));
end
