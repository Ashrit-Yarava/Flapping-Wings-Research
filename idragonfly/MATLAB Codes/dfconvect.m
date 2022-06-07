function [ZW]=dfconvect(ZF,VELF,dt,iGAMAf)
%Convect vortex from ZF to ZW using the velocity VELF*df
%INPUT
% ZF    location of the vortex
% VELF  velocity at the vortex
% dt    time interval
% iGAMAf    # of vortices to be convected
%OUTPUT
% ZW    location of the vortex after convection
    for iwing=1:2
    for i=1:iGAMAf(1)
        ZW(iwing,i)=ZF(iwing,i)+VELF(iwing,i)*dt;
    end
    end


end

