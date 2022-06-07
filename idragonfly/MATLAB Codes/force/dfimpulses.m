function [Limpb, Aimpb, Limpw,Aimpw ] = dfimpulses(ZVT,ZWT,GAM,m,GAMAw,iGAMAw)
%Calculate the linear and angular impulses on the airfoil in the body-fixed system
%INPUT
% ZVT       vortex points on theairfoil in the translating system
% ZWT       wake vortex location in the translating system
% GAM      bound vortices
% m         % of bound vortices
% GAMAw     wake vortices
% iGAMAw    # of wake vortices


%Bound vortex (free vortices excluded)
%Initialize the impulses before adding up
tmp=zeros(2,1);
Limpb=complex(tmp,tmp);
Aimpb=complex(tmp,tmp);
Limpw=complex(tmp,tmp);
Aimpw=complex(tmp,tmp);
    for I=1:m
        Limpb(:)=Limpb(:)-1i* GAM(:,I).*ZVT(:,I);
        Aimpb(:)=Aimpb(:)-0.5*GAM(:,I).*abs(ZVT(:,I)).^2;
    end
%Wake vortex 
    for I=1:iGAMAw(1)  %iGAMAw(1)=IGAMAW(2) %skips for istep=1 (iGAMAw=0)
        Limpw(:)=Limpw(:)-1i* GAMAw(:,I).*ZWT(:,I); 
        Aimpw(:)=Aimpw(:)-0.5*GAMAw(:,I).*abs(ZWT(:,I)).^2;
    end

end

