function [rT,v_,t_, d_,d,e,c,x,y,a,b,beta,delta,gMax,U,V ] = dfinData(l_,phiT_,phiB_,c_,x_,y_,a_,b_,beta_,delta_,f_,gMax_,U_,V_) 
%==========================================================================
%INPUT VARIABLES
% l_        span length (length)
% c_        chord length (length)
% x_,y_     airfoil data points (length)
% phiT_     top stroke angle (>0)
% phiB_     bottom stroke angle (<0)
% a_        rotation distance offset (length)
% b_        wing separation (length)
% f_        flap frequency
% beta_     stroke plane angle
% delta_    body angle (degree)
% gMax_     maximum rotation
% p         rotation velocity parameter (nondimentional)
% rtOff     rotation timing offset (nondimensional)
% U_,V_     ambient velocity
%==========================================================================
%OUTPUT
% d_        stroke length (dimentional)
% d         stroke length (nd); d(1)=1 but d(2)~=1
% rT        ratio of two periods (front/rear)
global fid
%Period
    T_=1.0./f_;
    fprintf(fid,'T_(1), T(2) = %6.3f %6.3f\n',T_);
    rT=T_(1)/T_(2);

%Covert angles to radian
    fac=pi/180.0;
    phiT=fac*phiT_;
    phiB=fac*phiB_;
    beta=fac*beta_;
    delta=fac*delta_;
    gMax=fac*gMax_;

%Nondimensional length quantities
    [v_,t_, d_,d,e,c,x,y,a,b, U, V] = dfndData(l_,phiT,phiB,c_,x_,y_,a_,b_,U_,V_,T_ );
   
end

