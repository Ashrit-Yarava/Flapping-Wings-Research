function [  ] = dfforceMoment(rho_,v_,d_1,nstep,dt ,U,V )
%Calculate the force and moment on the airfoil
%INPUT
% rho_  density of air
% v_    reference velocity
% d_1=d_(1)
% nstep     # of step
% dt        time increment

%OUTPUT (6/24/16)
%Fx_avr,Fy_avr, Mz_avr
global Limpulseb Aimpulseb Limpulsew Aimpulsew
global  folder fid

%Initialize force and moment array
    nwing=2;
    tmp=zeros(nwing,nstep);
    forceb =complex(tmp,tmp);
    forcew =complex(tmp,tmp);
    force  =complex(tmp,tmp);
    momentb=tmp;   
    momentw=tmp;    
    moment =tmp;
    temp=zeros(1,nstep);
    Tforce=complex(temp,temp);
    Tmoment=temp;
%Reference values of force and moment   
    f_=rho_*(v_^2)*d_1;
    m_=f_*d_1;
    
    for IT=1:nstep  
        U0 =complex(-U, -V); %velocity of the body translational system due to ambient velocity
        %U0 =complex(0.0, 0.0); %velocity of the body translational system due to its own motion
        for i=1:nwing
            if IT == 1 
                 forceb(i,1) = (Limpulseb(i,2)-Limpulseb(i,1))/dt;
                 forcew(i,1) = (Limpulsew(i,2)-Limpulsew(i,1))/dt;
                momentb(i,1) = (Aimpulseb(i,2)-Aimpulseb(i,1))/dt;
                momentw(i,1) = (Aimpulsew(i,2)-Aimpulsew(i,1))/dt;
               
            elseif IT == nstep
                 forceb(i,IT) = 0.5*( 3.0*Limpulseb(i,IT)-4.0*Limpulseb(i,IT-1)+Limpulseb(i,IT-2) )/dt;
                momentb(i,IT) = 0.5*( 3.0*Aimpulseb(i,IT)-4.0*Aimpulseb(i,IT-1)+Aimpulseb(i,IT-2) )/dt;
                 forcew(i,IT) = 0.5*( 3.0*Limpulsew(i,IT)-4.0*Limpulsew(i,IT-1)+Limpulsew(i,IT-2) )/dt;
                momentw(i,IT) = 0.5*( 3.0*Aimpulsew(i,IT)-4.0*Aimpulsew(i,IT-1)+Aimpulsew(i,IT-2) )/dt;
            else
                 forceb(i,IT) = 0.5*(Limpulseb(i,IT+1)-Limpulseb(i,IT-1))/dt;
                momentb(i,IT) = 0.5*(Aimpulseb(i,IT+1)-Aimpulseb(i,IT-1))/dt;
                 forcew(i,IT) = 0.5*(Limpulsew(i,IT+1)-Limpulsew(i,IT-1))/dt;
                momentw(i,IT) = 0.5*(Aimpulsew(i,IT+1)-Aimpulsew(i,IT-1))/dt;
            end   
       
            momentb(i,IT) = momentb(i,IT)+imag( conj(U0)*Limpulseb(i,IT) );
            momentw(i,IT) = momentw(i,IT)+imag( conj(U0)*Limpulsew(i,IT) );
       
    
            %Total Force and Moment (these are on the fluid)
            force(i,IT)  =  forceb(i,IT)+ forcew(i,IT);
            moment(i,IT) = momentb(i,IT)+momentw(i,IT);
        end
        %The dimensional force & moment on the wing are obtained by reversing the sign
        %and multiplying thr reference quantities
        Tforce(IT)  = -f_*(force(1,IT)+force(2,IT));
        Tmoment(IT) = -m_*(moment(1,IT)+moment(2,IT));
    end 
 
    ITa=linspace(1,nstep,nstep);
    fm=figure();
        plot(ITa,real(Tforce),'x-k');
        grid on;
        saveas(fm,[folder 'fx.tif']);
        %close;
        plot(ITa,imag(Tforce),'+-k');
        grid on;
        saveas(fm,[folder 'fy.tif']);
        %close;
        plot(ITa,Tmoment,'o-r');
        grid on;
        saveas(fm,[folder 'm.tif']);
    close;
%Calculate the average forces and moment
    Fx=real(Tforce);
    Fy=imag(Tforce);
    Mz=Tmoment;
    Fx_avr=sum(Fx)/nstep;
    Fy_avr=sum(Fy)/nstep;
    Mz_avr=sum(Mz)/nstep;
    fprintf(fid, 'Fx_avr = %6.3e Fy_avr = %6.3e Mz_avr = %6.3e\n', Fx_avr,Fy_avr, Mz_avr);


end

