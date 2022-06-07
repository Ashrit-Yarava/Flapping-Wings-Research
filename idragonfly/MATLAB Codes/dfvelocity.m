function [VEL]=dfvelocity(ZF,iGAMAf,GAM,m,ZV,GAMAw,iGAMAw)
%{
Calculates the velocity at free & wake vortex sites in th global system
Note that the ambient air speed is included, as its negative, in the wing
motion. Thus, the velocity contributions come only from the vortices.
%}
%INPUT
% ZF        sites of vortes to be convected & shed (global suystem)
%iGAMAf     # of vortices to be shed or convected
% GAMA      bound vortex
% m         # of bound vortices
% ZV     location of bound vortices (global)
% GAMAw     wake vortex
% iGAMAw    #of wake vortices
%OUTPUT
% VEL       velocity (not the conjugate) of vortex sites to be convected or shed
    for iwing=1:2
    for i=1:iGAMAf(1)
        VEL(iwing,i)=complex(0.0,0.0);
        for jwing=1:2
        for j=1:m           
            [VELF]=velVortex(GAM(jwing,j),ZF(iwing,i),ZV(jwing,j));
            VEL(iwing,i)=VEL(iwing,i)+VELF;
        end
        end
        for jwing=1:2
        for j=1:iGAMAw(1) %skips if iGAMw=0
            [VELF]=velVortex(GAMAw(jwing,j),ZF(iwing,i),ZF(jwing,j));
            VEL(iwing,i)=VEL(iwing,i)+VELF;
        end
        end
    end
    end
    


end

