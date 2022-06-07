
function [ y ] = dftableG(t,rt,tau,p,rtOff)
%Table function for an arbitrary time


tB=rem(t,2.0/rt);
[y]=dftableB(tB,rt,tau,p,rtOff);

end
