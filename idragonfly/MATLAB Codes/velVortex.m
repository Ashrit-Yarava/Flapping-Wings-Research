function [ v ] = velVortex( GAM, z, z0)
%Calculate the velocity at z due to vortex GAM at z0
%Use the vortex core model
%INPUT
% GAM   vortex
% z     destination
% z0    source
%OUTPUT 
% v     velocity complex(vx,vy)
global eps DELta ibios

%{
    v=complex(0.0,0.0);
    r=abs(z-z0);

    if r > eps
        v=-1i*GAM/(z-z0)/(2.0*pi);
    end
%}
r=abs(z-z0);
switch ibios       
    case 0   %Do not use the vortex core model
        eps=eps*1000; %Loosen the zero threshold radius
        v=complex(0.0,0.0);              %if too close, then set to zero
        if r > eps
            v=-1i*GAM/(z-z0)/(2.0*pi);
        end 
    case 1   %Use the vortex core model
        if r < eps
            v=complex(0.0,0.0);          %if too close, then set to zero
        else 
            v=-1i*GAM/(z-z0)/(2.0*pi);   %outside vortex core
            if r < DELta
            v=v*(r/DELta)^2;             %inside vortex core
            end
        end 
end
%Convert the complex velocity v=v_x-i*v_y to true velocity v=v_x+i*v_y
v=conj(v);

end

