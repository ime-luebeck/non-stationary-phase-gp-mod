function [ theta ] = timeWrap( t, tau )

N = length(tau);
phase = 2*pi*(0:(N-1));
theta = interp1(tau, phase, t, 'linear', 'extrap');

end

