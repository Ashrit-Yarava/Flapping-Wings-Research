function [v_, t_,d_,d,e,c,x,y,a, b,U, V] = dfndData(l_,phiT,phiB,c_,x_,y_,a_,b_,U_,V_,T_ )
global fid 
%Convert the insect flight input length parameters (dimensional) into
% non-dimensional parameters
%INPUT VARIABLES
% l_        span length (length)
% c_        chord length (length)
% x_, y_    airfoil data points
% phiT      top stroke angle (>0)
% phiB      bottom stroke angle (<0)
% a_        rotation distance offset (length)
% b_        wing location
% U_,V_     air speed (cm/sec)
% T_        period (sec)

%OUTPUT
% d_        stroke length (dimentional)
% v_
% t_

% Get nondimentional quantities 
% based on the given flight data of actual insect
    dT_=l_.*sin( phiT);
    dB_=l_.*sin(-phiB);
    d_=dT_+dB_;    %total stroke length
    fprintf(fid,'d_(1), d_(2) = %6.3f %6.3f\n',d_);
    e_=dT_-dB_;    %stroke difference
    d=d_/d_(1);   %d_(1) is the reference length       
    e=e_/d_(1);
    c=c_/d_(1);
    fprintf(fid,'c(1), c(2)  = %6.3f %6.3f\n',c);
    a=a_/d_(1);
    b=b_/d_(1);
    x=x_/d_(1);
    y=y_/d_(1);


%t_ = reference time  (Use the time for the front wing) 
    t_ = T_(1)/2.0;
%v_ = reference velocity (use the front wing flapping velocity)
    v_=d_(1)/t_;
    fprintf(fid,'v_ = %6.3f\n',v_);
%ambient velocity (nondimentional)
    U=U_/v_;
    V=V_/v_;
    fprintf(fid,'U = %6.3f V = %6.3f\n',U,V);

end

