function [VNW] = dfnVelocityw(m,ZC, NC, nwing,ZF, GAMAw, iGAMAw)
%Normal velocity contribution on the airfoil by the wake vortex
%INPUT
% m                         # of vortex points
% ZC        (1,m-1)         collocation points
% NC        (1,m-1)         unit normal complex number
% nwing     # of wings
% ZF        (nwing,iGAMAw)  location of the wake vortices (1:iGAMAw) sama sa ZW
% GAMAw     (nwing,iGAMAw)  wake vortex
% iGAMAw    (nwing)         # of wake vortices
%OUTPUT
% VNW       (1,m-1)         normal velocity components at the collocation point due to the wake vortices

global eps DELta ibios

VNW=zeros(1,m-1);
switch ibios
    case 0
        eps=eps*1000;
        for i=1:m-1
            VNW(i)=0.0;
            for iwing=1:nwing
                for j=1:iGAMAw(iwing) %skipped for iGAMAw=0 in istep=1          
                    r=abs(ZC(i)-ZF(iwing,j));
                    GF=complex(0.0,0.0);
                    if r > eps 
                        GF=1.0/(ZC(i)-ZF(iwing,j));
                    end
                    VNW(i)=VNW(i)+GAMAw(iwing,j)*imag(NC(i)*GF)/(2.0*pi);
                end
            end
        end
    case 1
        for i=1:m-1
            VNW(i)=0.0;
            for iwing=1:nwing
                for j=1:iGAMAw(iwing) %skipped for iGAMAw=0 in istep=1          
                    r=abs(ZC(i)-ZF(iwing,j));
                    if r < eps
                        GF=complex(0.0,0.0);
                    else 
                        GF=1.0/(ZC(i)-ZF(iwing,j));
                        if r < DELta
                            GF=GF*(r/DELta)^2;
                        end
                    end
                    VNW(i)=VNW(i)+GAMAw(iwing,j)*imag(NC(i)*GF)/(2.0*pi);
                end
            end
        end

end
%{
    for i=1:m-1
        VNW(i)=0.0;
        for iwing=1:nwing
            for j=1:iGAMAw(iwing) %skipped for iGAMAw=0 in istep=1
                GF=1.0/(ZC(i)-ZF(iwing,j));
                VNW(i)=VNW(i)+GAMAw(iwing,j)*imag(NC(i)*GF)/(2.0*pi);
            end
        end
%}
end

