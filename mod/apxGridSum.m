function [ K ] = apxGridSum( covs )

% if nargin==1                                        % report number of parameters
%   cov1 = covs{1};
%   cov2 = covs{2};
%   K = [feval(cov1{:}), '+', feval(cov2{:})];
%   %K = char(j(1)); for ii=2:length(cov), K = [K, '+', char(j(ii))]; end, return
% end

nc = numel(covs);                        % number of terms in covariance function
for ii = 1:nc                                % iterate over covariance functions
  f = covs(ii); if iscell(f{:}), f = f{:}; end   % expand cell array if necessary
  j(ii) = cellstr(feval(f{:}));                          % collect number hypers
end

if nargin==1  
    K = char(j(1)); for ii=2:nc, K = [K, '+', char(j(ii))]; end, return
end

end

