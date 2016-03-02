function y = ARMA_sample( constant, params, arma_part, n_cycle, varargin)
% This code is to generate time series history by an ARMA model
% the ARMA model is: yn= constant + phi1*y_(n-1)+...phi_p*y_(n-p) + e_n +
% theta1*e_(n-1) + ... + theta_q*e_(n-q) where the noise terms are iid of
% the zero mean Gaussian distribution N(0, sigma^2)
% constant is a scalar
% params = [phi_1 ... phi_p, theta_1 ... theta_q, sigma^2];
% arma_part = [p, q];
% n_cycle = number of times steps of the history
% varargin is to specify the initial seed to generate random samples
%% check
if length(params) ~= sum(arma_part) +1;
    disp('Error: params not consistent with arma_part');
    return;
end
%% set seed
if nargin>4;
    rng(varargin{1});
end
%% set initial max(p, q) values
p = arma_part(1);
q = arma_part(2);
k = max(p, q); % number of necessary initial values
burnin = 3*k;
y = zeros(n_cycle+burnin, 1);
cov_vector = transpose(arma_covs(k, params, arma_part)); % the row vector of first (k+1) covariance, [cov(0), cov(1), ..., cov(k)]
mu = constant/(1-sum(params(1:p)));
if k==1;
    cov_matrix = cov_vector(1);
    y(1:k, 1) = normrnd(mu, cov_matrix);
else
    cov_rep = repmat(cov_vector(1:k), [k, 1]); % k * k matrix
    cov_rep2 = [fliplr(cov_rep), cov_rep(:, 2:end)]; % k * (2k-1) matrix
    cov_matrix = spdiags(cov_rep2, -(k-1):(k-1), k, k); % the covariance matrix of the first k variables
    cov_matrix = full(cov_matrix);
    y(1:k, 1) = transpose(mvnrnd(repmat(mu, [1, k]), cov_matrix));
end
%% other values
e = normrnd(0, sqrt(params(end)), [length(y), 1]);
if p~=0 && q~=0; % ARMA
    for i = k+1:length(y);
        y(i, 1) = constant + dot(params(1:p), flipud(y(i-p:i-1))) + e(i) + dot(params(p+1:end-1), flipud(e(i-q:i-1)));
    end
elseif p~=0 && q==0; % AR
    for i = k+1:length(y);
        y(i, 1) = constant + dot(params(1:p), flipud(y(i-p:i-1))) + e(i);
    end
elseif p==0 && q~=0; % MA
    for i = k+1:length(y);
        y(i, 1) = constant + e(i) + dot(params(p+1:end-1), flipud(e(i-q:i-1)));
    end
end
%% burnin
y = y(burnin+1:end, 1);
end