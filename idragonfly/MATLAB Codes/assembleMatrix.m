function [ MVN ] = assembleMatrix( m,MVNs,MVNs_12,MVNs_21 )
%Assemblle 4 sub-matrices 
%INPUT
% MVNs      diagonal
% MVNs_12   off-diagonal
% MVNs_21   off-diagonal
    MVN(   1 :  m,    1 :  m)=MVNs(   1:m,1:m,1);
    MVN(   1 :  m, (m+1):2*m)=MVNs_12(1:m,1:m);
    MVN((m+1):2*m,    1 :  m)=MVNs_21(1:m,1:m);
    MVN((m+1):2*m, (m+1):2*m)=MVNs(   1:m,1:m,2);
end

