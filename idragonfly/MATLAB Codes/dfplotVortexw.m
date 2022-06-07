function [] = dfplotVortexw(istep,nwing,iGAMAw,ZV,ZW)
%Plot wake vortices in the space-fixed system
%INPUT
% iGAMAw(2)                # of wake vortices after shedding the free vortices
% ZV(2,m)                 vortex points on the airfoil
% ZW(2,2*nstep)       complex valued location in the space-fixed system

global folder wplot

%Airfoil
    XV  =real(ZV);
    YV  =imag(ZV);
    
%Wake
if istep ~= 1 %No wake for istep=1
    XW  =real(ZW);
    YW  =imag(ZW);    
end

%Plot and save to a  file
if wplot == 1
    
    
    if istep == 1
        fig=figure();
        %No wake vortex in istep=1.
        plot(XV(1,:), YV(1,:), '-k',XV(2,:), YV(2,:), '-k');    
        saveas(fig,[folder 'wake/' 'wake_' num2str(istep-1) '.tif']);
        close;
    else   
        iodd=1:2:iGAMAw(1)-1;
        ieven=2:2:iGAMAw(1);
        for i=1:nwing
            XWo(i,:)=XW(i,iodd);
            YWo(i,:)=YW(i,iodd);
            XWe(i,:)=XW(i,ieven);
            YWe(i,:)=YW(i,ieven);
        end
        fig=figure();
        %Plot front wing wake vortices from the leading/trailing edges black/red circles
        %     rear  wing wake vortices from the leading/trailing edges black/red x
        plot(XV(1,:), YV(1,:), '-k',XV(2,:), YV(2,:), '-k',...
            XWo(1,:),YWo(1,:), 'ok',XWe(1,:),YWe(1,:), 'or',...
            XWo(2,:),YWo(2,:), 'xk',XWe(2,:),YWe(2,:), 'xr');    
        saveas(fig,['fig/wake/wake_' num2str(istep-1) '.tif']);
        %print(gcf,'-dbmp',['fig/wake/' 'wake_' num2str(t) '.bmp']);
        close;
    end
   
end
end

