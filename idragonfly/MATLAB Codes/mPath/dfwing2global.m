function [NC,ZV,ZC,ZVb,ZCb,ZWb] = dfwing2global(istep,t,a,alp,l,h,xv,yv,xc,yc,dfc,ZW,U,V)
%Get global position of the wing colllocation and vortex points
%given coordinates in the wing-fixed coordinate system
%INPUT Variables (all nondimentional)
% t         time
% e         stroke difference
% a         rotation distance offset 
% alp       alpha
% l, h      location of the origin of the translating system
% xv, yv    coordinates of the vortex points in the wing-fixed system
% xc, yc    coordinates of the collocation points in the wing fixed system
% dfc       slope at the collocation points in the wing-fixed system
% ZW        wake vortex in the global system
% U, V      ambient velocity
%OUTPUT
% ZV,ZC         complex coordinates of the wing vortex and collocation points in the global system
% ZVb,ZCb,ZWb   complex coordinates of the wing vortex and collocation & wake vortex points  in the translational system
% NC            complex unit normal at collocation points in the global system
%= =========================================================================
global iplot nplot folder

    zb=complex(-U*t,-V*t);
    ZWb=ZW; %for istep=1, ZW is assigned to the initial zero value 
    if istep ~= 1
        ZWb=ZW-zb;
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
    %Position in the body translating system
    ZVb=ZV-zb;
    ZCb=ZC-zb;
 
    %Unit normal vector of the airfoil in the wing-fixed system
    denom=sqrt(1+dfc.^2);
    nx=-dfc./denom;
    ny= 1.0./denom;
    nc=complex(nx,ny);
    %Unit normal vector of the airfoil in the global system
    NC=nc*expmia;
       
    
    if iplot == 1
        plot(real(ZC),imag(ZC));
        hold on;
        plot(x0,z0,'or');
        axis equal
    end    
       
    if nplot == 1
        f=figure();
        plot(real(ZC),imag(ZC),'o');
        hold on;
        axis equal;    
        %End points for the unit normal vector at collocation points
        sf=0.025; %scale factor for the  plot
        xaif=real(ZC);
        yaif=imag(ZC);
        xtip=xaif+sf*real(NC);
        ytip=yaif+sf*imag(NC);
        %Plot unit normal vectors at collocation points     
        plot([xaif; xtip],[yaif; ytip])
        hold off   
        saveas(f,[folder,'w2g_',num2str(t),'.tif']);
        close;
    end
end

