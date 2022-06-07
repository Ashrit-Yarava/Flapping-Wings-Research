function [xle,zle,xte,zte  ] = dfchordPath(l,h,alp,a,c)
%INPUT Variables (all nondimentional)

% l         horizontal coordinate of the translating wing system
% h         vertical coordinate of the translating wing system
% a         rotation distance offset 
% c     chord length
% alp       attack angle

%==========================================================================
%LOCAL Variables
    
%Edge positions for the composite motion
    xle=l+(a-0.5*c)*cos(alp);
    zle=h-(a-0.5*c)*sin(alp);
    xte=l+(a+0.5*c)*cos(alp);
    zte=h-(a+0.5*c)*sin(alp);

end

