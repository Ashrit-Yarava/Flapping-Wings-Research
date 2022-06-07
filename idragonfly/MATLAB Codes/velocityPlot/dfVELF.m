function [ VVspace] = dfVELF(Z,ZV,ZW, GAM,m,GAMAw,iGAMAw)    
%--------------------------------------------------------------------------
%Calculation of the velocity fields VVspace (wing)=[u v] using the wing-fixed mesh ZETA
%INPUT
% xv,yv     bound vortex location
% zw        wake vortex location
% a         rotation axis offset
% GAM      bound vortex
% m         # of bound vortex
% GAMAw     wake vortex
% iGAMAw    # of wake vortices
% U,V       free stream velocity
% alp       airfoil rotation
% dalp,dl,dh    airfoil velocity components
%OUTPUT:
% VVspace   complex velocity in the space-fixed system
% VVwing    complex velocity in the wing-fixed system
%--------------------------------------------------------------------------


% calculate velocity at ZETA mesh points
%    for i=1:iGAMAf
%        vel(i)=complex(0.0,0.0);
%        for j=1:m           
%            [velf]=velVortex(GAMA(j),zf(i),zv(j));
%            vel(i)=vel(i)+velf;
%        end
%        for j=1:iGAMAw %skips if iGAMw=0
%            [velf]=velVortex(GAMAw(j),zf(i),zf(j));
%            vel(i)=vel(i)+velf;
%        end
%    end
    
%Initialize the Complex velocity at 
    sz=size(Z);
    VV=complex(0,0)*ones(sz); 
    
%Contribution from the bound vortices 
    for iwing=1:2
    for J=1:m
        for i=1:sz(1)
        for j=1:sz(2)
            VV(i,j)=VV(i,j)+velVortex(GAMA(iwing,J),Z(i,j),ZV(iwing,J));
        end
        end
    end
    end
%Contribution from the wake vortex 
    for iwing=1:2
    for J=1:iGAMAw(1) %could use iGAMAw(2) 
        for i=1:sz(1)
        for j=1:sz(2)
            VV(i,j)=VV(i,j)+velVortex(GAMAw(iwing,J),Z(i,j),ZW(iwing,J));
        end
        end           
    end
    end
%VV is the real velocity
    %VV=conj(VV);
%Contribution from the free stream 
    VVspace=VV; 
   

    
end

