function [ZVFR,ZCFR,ZWFR] = dfwing2FRT(istep,a,alp,l,h,xv,yv,xc,yc,ZW,lFR,hFR)
%Get forward/rear wing translational position of the wing colllocation and vortex points
%REPEAT this function for two wings
%given coordinates in the wing-fixed coordinate system
%INPUT Variables (all nondimentional)

% istep
% a         rotation distance offset 
% alp       alpha
% l, h      location of the origin of the translating system 
% xv, yv    coordinates of the vortex points in the wing-fixed system
% xc, yc    coordinates of the collocation points in the wing fixed system
% dfc       slope at the collocation points in the wing-fixed system
% ZW        wake vortex in the global system
% lFR,hFR  forward/rear wing motion parameters

%OUTPUT
% ZVFR,ZCF,ZWF   complex coordinates of the wing vortex and collocation & wake vortex points  in the forward wing translational system
%= =========================================================================

    %Global coordinates of the forward wing translating center
    ztFR=complex(lFR,hFR);
    ZWFR=ZW; %for istep=1, ZW is assigned to the initial zero value 
    if istep ~= 1
        ZWFR=ZW-ztFR;
    end
    %Global coordinates of the current wing translating system origin
    zt=complex(l,h); 
    %Global positions for the collocation and vortex points on the wing
    %Add translational and rotational motion contributions
    zv=complex(xv,yv);
    zc=complex(xc,yc);
    expmia=exp(-1i*alp);
    ZVT=(a+zv)*expmia;      %Position in the translating system 
    ZCT=(a+zc)*expmia;      %Position in the translating system 
    ZV=ZVT+zt;
    ZC=ZCT+zt;
    %Position in the forward/rear wing translating system
    ZVFR=ZV-ztFR;
    ZCFR=ZC-ztFR;
 
end

