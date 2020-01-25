function [ SNR ] = get_SNR( sig, indices1, indices2, winlen1, winlen2, fs)

% GET_SNR
% 
% Description:
%   Estimate the SNR as the ratio of signal energy around the R peak as
%   compared to between the R peaks.
%
% References:
%   Bartolo, A., Roberts, C., Dzwonczyk, R. R., and Goldman, 
%   E. Analysis of diaphragm emg signals: comparison of
%   gating vs. subtraction for removal of ecg contamination.
%   Journal of Applied Physiology, 80(6):1898–1902, 1996.

gates1 = zeros(size(sig));
gates2 = zeros(size(sig));

half_win_samples1 = round(winlen1*fs/2);
half_win_samples2 = round(winlen2*fs/2);

for i=1:length(indices1)
    gates1(max(1, indices1(i) - half_win_samples1) : min(length(sig), indices1(i) + half_win_samples1)) = 1;
end

for i=1:length(indices2)
    gates2(max(1, indices2(i) - half_win_samples2) : min(length(sig), indices2(i) + half_win_samples2)) = 1;
end

gates1_without_gates2 = max(0, gates1 - gates2);
gates2_without_gates1 = max(0, gates2 - gates1);

sig1_gated = sig;
sig1_gated(gates1_without_gates2==0) = 0;
sig2_gated = sig;
sig2_gated(gates2_without_gates1==0) = 0;

%rms1 = sqrt(mean(sig(gates1_without_gates2==1).^2));
%rms2 = sqrt(mean(sig(gates2_without_gates1==1).^2));

rms1 = sqrt(mean(sig1_gated.^2));
rms2 = sqrt(mean(sig2_gated.^2));

SNR = rms2/rms1;

end

