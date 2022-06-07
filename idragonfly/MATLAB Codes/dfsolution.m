function [ GAMA ] = dfsolution(m,MVN,VN,VNW,sGAMAw)
%Solution
%INPUT
% istep         time step
% m             # of bound vortices for each wing
% VN(2,m-1)     normal velocity at the collocation points (m-1 components) by the bound vortex
% VNW(2,m-1)    normal velocity at the collocation points (m-1 components) by the wake vortex
% sGAMAw(2)     sum of the wake vortices
%OUTPUT
% GAMA  bound vortices
%RHS of the equation
%FORWARD WING
    %for the non-peneteration condition: (m-1) components 
    GAMA(  1  :(  m-1))=VN(1,1:(m-1))-VNW(1,1:(m-1));
    %for the vortex conservation condition: 1 component 
    GAMA(  m)=-sGAMAw(1);
%REAR WING
    %for the non-peneteration condition: (m-1) components
    GAMA((m+1):(2*m-1))=VN(2,1:(m-1))-VNW(2,1:(m-1));
    %for the vortex conservation condition: 1 component
    GAMA(2*m)=-sGAMAw(2);
    
    [ip,MVN]= DECOMP(2*m,MVN);
    [GAMA]= SOLVER(2*m,MVN,GAMA,ip);    
end

