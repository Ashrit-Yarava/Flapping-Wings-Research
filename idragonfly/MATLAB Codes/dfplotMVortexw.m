function [  ] = dfplotMVortexw(v_,d_1, GAMAw,nstep )
%Print and plot the magnitudes of the wake vortex
%INPUT
% d_1=d_(1)
% GAMAw     wake vortex
% nstep     # of time steps

%Reference value for the circulation
    nwing=2;
    gama_=v_*d_1;   
    iodd=1:2:2*nstep-1;
    ieven=2:2:2*nstep;
    %Dimensional values of the circulation
    for i=1:nwing
        it=1:nstep;
        GAMAwo(i,it)=gama_*GAMAw(i,iodd);
        GAMAwe(i,it)=gama_*GAMAw(i,ieven);
        f=figure();
        plot(it, GAMAwo(i,it), 'o-k', it,GAMAwe(i,it), 'o-r');
        grid on;
        saveas(f,['fig/GAMAw_(' num2str(i) ').tif']);
        close;
    end
    
end

