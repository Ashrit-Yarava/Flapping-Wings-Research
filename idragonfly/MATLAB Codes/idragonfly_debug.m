function [  ] = idragonfly_debug( )
%Two wings flapping independeltly
%Thin 2D airfoil by the discrete vortex method is used
%Motion path consists of the flapping and rotation, which is specified 
%   based on the dragonfly morphology and flight data.
%Formulate in the space-fiee coordinate system based on igVortex.m
%Version 1, October, 23, 2014 by Mitch Denda
%   impulses forces and moment is calculated in the body-translating system
%Version 2, November 23, 2015
%   Automatic determination of time increment based on m and p
%   Adopt vortex core model
%==========================================================================
%GLOBAL VARIABLES
global mplot vplot fid eps folder wplot zavoid DELta ibios
global svCont wvCont  ivCont vpFreq

%Impulses wrt body-translating system
global Limpulseb Aimpulseb Limpulsew Aimpulsew LDOT HDOT ZETA
%Impulses wrt forward wing translating system
global LimpulsebF AimpulsebF LimpulsewF AimpulsewF
%Impulses wrt rear wing translating system
global LimpulsebR AimpulsebR LimpulsewR AimpulsewR

%File manupulation=========================================================
%Open an Output File:    
fid=fopen([folder 'output.txt'],'w');
%How to specify the file path:
    folder = 'fig/';
%==========================================================================
%Time specification: stime 0(single time)  2(time marching)
%3(multiple time for chord path plot)
    stime = 2; %Use this in the production phase
    %stime = 3;     %For chord path plot
    %stime = 0;     %For performance test of functions in istep=1
    
%DEBUGGING PARAMETERS======================================================

%Airfoil Mesh plot: mplot mesh plot 0 (no), 1(yes), 2(Compare equal arc and equal abscissa mesh points)
    mplot = 0; %Keep this in the production phase
%Airfoil normal velocity plot: vplot 0(no), 1(yes)
    vplot = 0; %Keep this in the production phase
%Wake Vortex plot: 0(n0), 1(yes)
    wplot = 1;  %Keep this for the production run
%Velocity plot by avoiding source and observation points coincidence:
%zaoid: 0 (no, faster), 1(yes, slower)
    zavoid = 0; 
    %zavoid = 1 %Use If the vllocity plot shows blowup, somewhere 
    
%Velocity field plot: 0 (no), 1 (yes)
    vfplot=0; %Use this in the production phase to save memory and sturage
    %vfplot = 1: % Use only for selective cases to be shown to the audience
%Around which wing to plot the velocity field: 1 (forewing), 2(rear wing)
    vplotFR=1;  %The original meshing is done in the local wing-fixed system for the forward wing
    %vplotFR=2;  %The original meshing is done in the local wing-fixed system for the rear wing    
%Velocity contour plots in space-fixed system
%Used as the 4th argument of 'contourf' to control the range of velocity
    %space-fixed velocity plot: svInc (increment), svMax (max velocity)
    svInc=0.025;
    svMax=2.50;
    svCont=0.0:svInc:svMax;
    %wing-fixed  velocity plot: wvInc (increment), wvMax (max velocity)
    wvInc=0.1;
    wvMax=7.0;
    wvCont=0:wvInc:wvMax;
%Use of svCont and wvCont: ivCont = 0 (no), 1 (yes)
%   The velocity range varies widely depending on the input parakmeters
%   It is recommended to respecify this when input parameters are changed
    ivCont = 0;
%Frequency of velocity plots: vpFreq
    vpFreq=1;
%END DEBUGGING PARAMETERS==================================================
       
%INPUT VARIABLES===========================================================
%==========BODY GEOMETRY
    nwing=2; %# of wings
%Body angle (degree)
    delta_=0;
%Wing location (cm): (1) fore wing, (2) rear wing
    b_(1)= -0.75; b_(2)=0.75; %Make sure b_(1)<=0, b_(2)>=0
    %b_(1)= -0.0; b_(2)=0.0; %Use this for clap and fling
%==========WING GEOMETRY
% l_ = wing span (cm)
    l_(1)  = 4; l_(2) = 4;
% c_ = chord length (cm)
%    calculated whie specifying the airfoil shape
%# of data points that define the airfoil shape
    n=101;
%Read airfoil shape data and determine the chord length
    %Here, use a formula to spefify the airfoil shape
    atmp_(1) =0.5;  atmp_(2) =0.5;
    camber(1)=+0.0; camber(2)=+0.0;
    for i=1:nwing
        x_(i,:)=linspace(-atmp_(i),atmp_(i),n);
        %Camber options are not ellaborated yet  
        y_(i,:)=camber(i)*(atmp_(i)^2-x_(i,:).^2);
        %OR give arrays x_ and y_ from NASA airfoil data base
        c_(i)=x_(i,n)-x_(i,1);
        fprintf(fid,'i = %3d c_ = %6.3f camber = %6.3f\n',i,c_(i),camber(i));
    end
%# of vortex points on the airfoil
    m=5;
    
%==========WING MOTION PARAMETERS
% stroke angles (degrees)
    phiT_(1)   = 45;    phiT_(2)   = 45;
    phiB_(1)   = -45;   phiB_(2)   = -45;
% a_ rotation axis offset (cm)
    a_(1)      = 0;     a_(2)      = 0;
% beta = stroke plane angle (degrees) wrt the body axis
    beta_(1)   = 30;    beta_(2)   = 30;
% f_ = flapping frequency (1/sec)
    f_(1)      = 30;    f_(2)      = 30;
% gMax = max rotation (degrees) amplitude: actual rotation is 2*gMax
    gMax_(1)   = 30;    gMax_(2)   = 30; 
    %gMax_(1)   = 15;    gMax_(2)   = 60;
    

% p = rotation speed parameter (nondimentional): p(i)=p_(i)/(0.5*T_(i))
% (Note p(1)=p_(1)/t_ref, but p(2) ~=(not equal)p_(2)/t_ref, wherere t_ref is the rference time)
%   p>=4
    p(1)       = 5;     p(2)       = 5;
    if p(1)<4
        disp('p(1) must be bigger than 4')
    end
    if p(2)<4
        disp('p(2) must be bigger than 4')
    end
% rtOff = rotation timing offset (nondimentional): rtOff(i)=rtOff_(i)/(0.5*T_(i))
% (Note rtOff(1)=rtOff_(1)/t_ref, but rtOff(2)~=rtOff_(2)/t_ref, wherere t_ref is the rference time)
%   rtOff<0(advanced), rtOff=0 (symmetric), rtOff>0(delayed)
%   -0.5<rtOff<0.5
    rtOff(1)   = 0.0;    rtOff(2)   = 0.0; %rtOff(1)=0.2
    if abs(rtOff(1))>0.5
        disp('-0.5<=rtOff(1)<=0.5 is not satisfied')
    end
    if abs(rtOff(2))>0.5
        disp('-0.5<=rtOff(2)<=0.5 is not satisfied')
    end
% tau = phase shift for the time: tau(i)=tau_(i)/(0.5*T_(i))
% (Note tau(1)=tau_(1)/t_ref, but tau(2)~=tau_(2)/t_ref, wherere t_ref is the rference time)
%   0(start from TOP and down), 0<tau<1(in between, start with DOWN STROKE),
%   1(BOTTOM and up), 1<tau<2(in between, start with UP STROKE), 2(TOP): 
%   0 <= tau < 2
    tau(1)     = 0.0;     tau(2)     = 1;
    if  tau(1) < 0 || tau(1) >2
        disp('0<= tau(1) < 2 is not satisfied')
    end
    if  tau(2) < 0 || tau(2) >2
        disp('0<= tau(2) < 2 is not satisfied')
    end
%Motion path parameter: mpath 0(no tail), 1 (DUTail; 2eriods), 2(UDTail; 2 periods),
%3(DUDUTail; 4 periods), 4(UDUDTail; 4 periods)
    mpath(1) = 0; mpath(2) = 0; %Other options are not implemented yet
    fprintf(fid,'mpath(1), mpath(2) = %3d %3d\n',mpath);    
%==========FLUID PARAMETERS
%Air density
    rho_=0.001225; %g/cm^3
%ambient velocity (cm/sec, assume constant)
%Can be interpreted as the flight velocity when the wind is calm
    U_= 100.0;
    V_= 0.0;


%Time increment and # of time steps option 0(manually specify),%1(automatic, recommended)
    %itinc=0; %manually specify the time increment
    itinc=1;  %Specify nperiod (# of periods) below
    
%Distance between the source and the observation point to be judged as zero
    eps=0.5E-06;  
%Vortex core model (Modified Biot-Savart equation): 0(no), 1(yes)
    ibios=1;
    fprintf(fid, 'ibios = %3d\n',ibios);    
    

%==========================================================================

%PRINT INPUT DATA==========================================================
for i=1:nwing
    fprintf(fid, 'iwing = %3d, l_(i)   = %6.3f, phiT_(i)= %6.3f, phiB_(i) = %6.3f, a_(i) = %6.3f, beta_(i) = %6.3f, f_(i) = %6.3f\n', ...
            i,l_(i),phiT_(i),phiB_(i),a_(i),beta_(i),f_(i));
    fprintf(fid, 'iwing = %3d,gMax_(i)= %6.3f, p(1)    = %6.3f, rtOff(i) = %6.3f, tau(i) = %6.3f\n',...
            i,gMax_(i),p(i),rtOff(i),tau(i));
end
    fprintf(fid, 'U_   = %6.3f, V_   = %6.3f, m     = %4d, n = %4d\n', ...
            U_,V_,m,n);
    %fprintf(fid,'nstep = %4d dt = %6.3f\n',nstep,dt);
%==========================================================================

%Nondimentionalize the input variables
    [rT,v_,t_, d_,d,e,c,x,y,a,b,beta,delta,gMax, U, V ] = dfinData(l_,phiT_,phiB_,c_,x_,y_,a_,b_, ...
                                           beta_,delta_,f_,gMax_,U_,V_);
    %period ratio: rt(i)=T_(1)/T_(i)
    rt(1)=1.0;  %
    rt(2)=rT;   %rT=T_(1)/T_(2)
    
    
    
%Threshold radius for modified Biot-Savart equation
    DELTA=0.5*c/(m-1);  %distance between the collocation point and the vortex point on the wing
                        %not to be confused with the body angle delta
    q=1.0;   %multiplier 0 < q <= 1
    DELTA=q*DELTA;
    DELta=min(DELTA);   %select smaller value od DELTA
    fprintf(fid,'q = %6.3f, delta = %6.3f\n',q, DELta);
%Time increment
switch itinc
    case 0 %Manual selection
        dt = 0.1;       %0.025 (for m=21)
        nstep = 21;     %81    (for m=21)
    case 1 %Automatic
        nperiod=1; %# of period to calculate (default=1)
        %select smaller of dtC=c(i)/(m-1), and the smaller of dtP=0.1*(4/p(i)
        %and further smaller of dtC and dtP.
        dt= min(min(c/(m-1), 0.1*(4./p))); %4/p = duration of pitch
        nstep=nperiod*ceil(2/dt);  % One period=2(nondimensional)
end
fprintf(fid,'nstep = %4d dt = %6.3f\n',nstep,dt);     
     
%Comparison of flapping, pitching and air speeds 
    air=sqrt(U_^2+V_^2);
    fprintf(fid, 'air speed = %6.3e\n',air);
    if air > 1.0E-03
        %Flapping/Air Seed Ratio
        fk=2*f_.*d_/sqrt(U_^2+V_^2);
        fprintf(fid, 'flapping/air: speed ratio = %6.3e %6.3e\n', fk);
        %Pitch/Flapping Speed Ratio
        r=0.25*(c_./d_).*(p/t_).*(gMax./f_);
        fprintf(fid, 'pitch/flapping: speed ratio = %6.3e %6.3e\n', r);
        %Pitch/Air Speed Ratio
        k=fk.*r;
        fprintf(fid, 'pitch/air: speed ratio = %6.3e %6.3e\n', k);
    else
        %Pitch/Flapping Speed Ratio
        r=0.25*(c_./d_).*(p/t_).*(gMax./f_);
        fprintf(fid, 'pitch/flapping: speed ratio = %6.3e %6.3e\n', r);        
    end
                                           
%Generate the vortex and collocation points on the airfoil 
    for i=1:nwing
        [xv(i,:), yv(i,:), xc(i,:), yc(i,:), dfc(i,:),mNew] = dfmeshR(c(i),x(i,:),y(i,:),n,m); %m is increased by 4 due to refinement
    end
    m=mNew;

%INITIALIZATION    
    %Initialize the wake vortex magnitude array
    % GAMAw(i,1:2) step 1, GAMAw(i,3:4) step 2, GAMAw(i,5:6) step 3, ...
    % Leading edge: odd components, Trailing edge: even components
    GAMAw=zeros(nwing,2*nstep);
    %Initialize the free vortex magnitude array
    %This is the vortex to be shed or convected
    %GAMAf=zeros(2,2*nstep);
    %Initialize the total wake vortex sum
    sGAMAw = zeros(1,nwing);
    %Initialize the total wake vortex number
    iGAMAw = zeros(1,nwing);
    %Initializing the # of vortices to be convected or shed
    iGAMAf = zeros(1,nwing);
    %Initialize the free+wake vortex location array (before convection)
    % ZF(i,1:2) step 1, ZF(i,3:4) step 2, ZF(i,5:6) step 3, ...
    % Leading edge: odd components, Trailing edge: even components
    temp = zeros(nwing,2);
    ZF = complex(temp,temp);
    %Initialize the wake vortex location array (after convection)
    % ZW(i,1:2) step 1, ZW(i,3:4) step 2, ZW(i,5:6) step 3, ...
    % Leading edge: odd components, Trailing edge: even components
    ZW = complex(temp,temp);
    %This is further transformed into a new body-fixed coordinate system
    
    %Initialize the linear and angular impuse array
    tmp=zeros(nwing,nstep);
    Limpulseb=complex(tmp,tmp) ; Aimpulseb=tmp ; Limpulsew=complex(tmp,tmp) ; Aimpulsew=tmp;
    LimpulsebF=complex(tmp,tmp); AimpulsebF=tmp; LimpulsewF=complex(tmp,tmp); AimpulsewF=tmp;
    LimpulsebR=complex(tmp,tmp); AimpulsebR=tmp; LimpulsewR=complex(tmp,tmp); AimpulsewR=tmp;
    %Initialize the translational velocity of F/R wings
    LDOT=zeros(nwing,nstep); HDOT=zeros(nwing,nstep);
    
    %Normal velocity on the wing due to the wing motion & wake vortices
    VN =zeros(nwing,m-1);
    VNW=zeros(nwing,m-1);
    %sub-matrix for the non-penetration condition matrix
    MVNs=zeros(m,m,nwing);
switch stime
case 0 %Check the performance of functions for istep=1        
        %%%CALCULATION OF MVNs_11=MVNs(:,:,1) and MVNs_22=MVNs(:,:,2)
        %Use the wing-fixed coordinate system to calcuate the sub-matrix coefficients. 
        %The sub-matrix coefficients obtained using the global system are identical to
        %these and remain constant throughout the time steps.

        for i=1:nwing
        [MVNs(:,:,i)]=selfMatrixCoef( xv(i,:),yv(i,:),xc(i,:),yc(i,:),dfc(i,:),m); %Calcuate only once and keep using for eaxh step
        end
        istep=1;
        t=(istep-1)*dt; %Use (istep-1) to start with time=0
        %Get airfoil motion parameters
        for i=1:nwing          
            [alp(i),l(i),h(i),dalp(i), dl(i),dh(i)] = dfairfoilM(mpath(i),t,rt(i),tau(i),d(i),e(i),b(i),beta(i),delta,gMax(i),p(i),rtOff(i),U,V);
            LDOT(i,istep)=dl(i);
            HDOT(i,istep)=dh(i);
        end
 
        %Wing votex(V)/collocation(C) points & wake(W) vortex points
        % ZV(i,:), ZC(i,:), ZW(i,:)    (global system)
        % ZVb(i,:),ZCb(i,:),ZWb(i,:)   (body translating system)
           % ZW in istep=1 is assigned zero (or null) by initialization
           % ZW=ZF for istep >=2 (see the last command of the time marching loop)       
        %Unit normal of the wing at the collocation points (global): NC(i,:)                      
        %Body translating system : This is the default system for the calculation of force and moment
        for i=1:nwing
            [NC(i,:),ZV(i,:),ZC(i,:),ZVb(i,:),ZCb(i,:),ZWb(i,:)]...
                = dfwing2global(istep,t,a(i),alp(i),l(i),h(i),xv(i,:),yv(i,:),xc(i,:),yc(i,:),dfc(i,:),ZW(i,:),U,V);
        end
        %wing translational systems for force moment calculation
            % ZVF(i,:),ZCF(i,:),ZWF(i,;)   (forward wing translating system)
            % ZVR(i,:),ZCR(i,:),ZWR(i,:)   (rear    wing translating system)
            % ZCF(1,:) is needed for the forward wing velocity calculation at the collocation points
            % ZCR(2,:) is needed for the rear    wing velocity calculation at the collocation points
            %Forward wing translating system
            for i=1:nwing
            [ZVF(i,:),ZCF(i,:),ZWF(i,:)]...
                = dfwing2FRT(istep,a(i),alp(i),l(i),h(i),xv(i,:),yv(i,:),xc(i,:),yc(i,:),ZW(i,:),l(1),h(1));
            end
            %Rear wing translating system
            for i=1:nwing
            [ZVR(i,:),ZCR(i,:),ZWR(i,:)]...
                = dfwing2FRT(istep,a(i),alp(i),l(i),h(i),xv(i,:),yv(i,:),xc(i,:),yc(i,:),ZW(i,:),l(2),h(2));
            end
              
        %Normal Velocity on the airfoil due to its motion
            %for iwing=1
            [VN(1,:)] = dfairfoilV(1,ZC(1,:),ZCF(1,:),NC(1,:),t,dl(1),dh(1),dalp(1));
            %for iwing=2
            [VN(2,:)] = dfairfoilV(2,ZC(2,:),ZCR(2,:),NC(2,:),t,dl(2),dh(2),dalp(2));    

        %Normal velocity on the airfoil due to the wake vortex
        for iwing=1:nwing
            [VNW(i,:)] = dfnVelocityw(m,ZC(i,:),NC(i,:),nwing,ZF,GAMAw,iGAMAw); %produces zero VNW for step 1, %ZF=ZW                                                                         
        end
        
        %Calculation of the sub-matrices MVNs_12 and MVNs_21: they are time dependent
        [MVNs_12]=crossMatrixCoef( ZV(2,:),ZC(1,:),dfc(1,:),m);
        [MVNs_21]=crossMatrixCoef( ZV(1,:),ZC(2,:),dfc(2,:),m);
        
        %Assemble the total matrix using MVNs(:,:,2), MVNs_12(:,:),MVNs_21(:,:)
        [ MVN ] = assembleMatrix( m,MVNs,MVNs_12,MVNs_21 );
        %Solve the system of equations
        [ GAMA ] = dfsolution(m,MVN,VN,VNW,sGAMAw);
        %Split GAMA into two parts
        GAM(1,1:m)=GAMA   (1 :  m); %Forward wing
        GAM(2,1:m)=GAMA((m+1):2*m); %Rear wing
                
        %Plot locations, ZW, of the wake vortices for the current step
        %in the space-fixed system
        dfplotVortexw(istep,nwing,iGAMAw,ZV,ZW);
        
        %Calculate the linear and angular impiulses on the airfoil=========
        %Include all of the bound vortices and wake vortices.
        %For istep=1, there is no wake vortices.
        
        %Use the body-translating system 
        [Limpb,Aimpb,Limpw,Aimpw] = dfimpulses(ZVb,ZWb, GAM,m,GAMAw,iGAMAw);
        Limpulseb(:,istep)=Limpb(:);
        Aimpulseb(:,istep)=Aimpb(:);
        Limpulsew(:,istep)=Limpw(:);
        Aimpulsew(:,istep)=Aimpw(:);      
        %Use the fore wing-translating system 
        [LimpbF,AimpbF,LimpwF,AimpwF] = dfimpulses(ZVF,ZWF, GAM,m,GAMAw,iGAMAw); 
        LimpulsebF(:,istep)=LimpbF(:);
        AimpulsebF(:,istep)=AimpbF(:);
        LimpulsewF(:,istep)=LimpwF(:);
        AimpulsewF(:,istep)=AimpwF(:); 
        %Use the rear wing-translating system 
        [LimpbR,AimpbR,LimpwR,AimpwR] = dfimpulses(ZVR,ZWR, GAM,m,GAMAw,iGAMAw);
        LimpulsebR(:,istep)=LimpbR(:);
        AimpulsebR(:,istep)=AimpbR(:);
        LimpulsewR(:,istep)=LimpwR(:);
        AimpulsewR(:,istep)=AimpwR(:);    
      
case 3 %PLOTTING THE CHORD PATH animation
    for istep =1:nstep
        t=(istep-1)*dt; %Use (istep-1) to start with time=0
        %Get airfoil motion parameters
        for i=1:nwing          
            [alp(i),l(i),h(i),dalp(i), dl(i),dh(i)] = dfairfoilM(mpath(i),t,rt(i),tau(i),d(i),e(i),b(i),beta(i),delta,gMax(i),p(i),rtOff(i),U,V);
        end
        for i=1:nwing
            [xle(i),zle(i),xte(i),zte(i)  ]                     = dfchordPath(l(i),h(i),alp(i),a(i),c(i));
            [xl(i,istep),zl(i,istep),xt(i,istep),zt(i,istep)  ] = dfchordPath(l(i),h(i),alp(i),a(i),c(i));
        end
        
        f=figure();
        plot([xle; xte],[zle; zte]);
        %axis([-0.75 0.75 -0.75 0.75]);
        axis([-1.5 1.5 -1.5 1.5]);
        axis equal;
        M(istep)=getframe;
        %saveas(f,[folder 'chordPath/chord_'  num2str(istep-1)  '.tif']);
        close;       
    end 
    %Play the movie and save as an avi file
    movie(M,3,10); %repeat 3 times at 10 fps
    movie2avi(M, 'fig/chordPath/dragonfly.avi', 'compression', 'none');
    %Plot he whole positions of the chords
    f=figure();
        plot([xl(1,:); xt(1,:)],[zl(1,:); zt(1,:)],'k');
        %axis([-1.5 1.5 -1.5 1.5]);
        axis equal
        hold on
        %saveas(f,[folder 'chord'  'Path_F'  '.tif']);       
        plot([xl(2,:); xt(2,:)],[zl(2,:); zt(2,:)],'r')
        axis equal
        saveas(f,[folder 'chordPath/chordPath_FR'  '.tif']);
    close;

case 2 
%time marching        
    %Vortex convection Time history sample (for each wing; index i is omitted)
    %For examplem use GAMA_1(1), ZV_1(1) instead of GAMA_1(i,1), ZV_1(i,1)
    %
    %step 1: iGAMAw_1=0, iGAMAf_1=2

     %GAMAw_1 = [0         , 0          ]; no wake vortex
     %GAMAf_1 = [GAMA_1(1) , GAMA_1(m)  ]; vortex to be convected or shed
     %ZF_1  = [ZV_1(1), ZV_1(m) ] = [ZF(1), ZF(2)]; leading and trailing edges
     %ZW_1  = [ ZW_1(1)  ,  ZW_1(2)   ]; Convect ZF_1 
        
    %step 2: iGAMAw_2=2, iGAMAf_2=4
     %GAMAw_2=GAMAf_1=[GAMA_1(1) , GAMA_1(m) ]; wake vortex
     %GAMAf_2=[GAMA_1(1) , GAMA_1(m), GAMA_2(1) , GAMA_2(m) ]; vortex to be convected or shed
     %ZF_2  = [ZW_1(1)  , ZW_1(2) , ZV_1(1), ZV_1(m) ] 
     %      = [ ZF_2(1)  ,  ZF_2(2) ,  ZF_2(3)  ,  ZF_2(4)  ]
     %ZW_2  = [ ZW_2(1)  ,  ZW_2(2) ,  ZW_2(3)  ,  ZW_2(4)  ]; Convect ZF_2 in the current coord system
        
    %step 3: iGAMAw_3=4, iGAMAf_3=6
     %GAMAw_3=GAMAf_2=[GAMA_1(1) , GAMA_1(m), GAMA_2(1) , GAMA_2(m) ]; wake vortex
     %GAMAf_3=[GAMA_1(1) , GAMA_1(m), GAMA_2(1) , GAMA_2(m), GAMA_3(1) , GAMA_3(m) ]; vortex to be convected or shed
     %ZF_3  = [ZW_2(1)  , ZW_2(2) , ZW_2(3)  , ZW_2(4) , ZV_1(1), ZV_1(m) ]
     %      = [ ZF_3(1)  ,  ZF_3(2) ,  ZF_3(3)  ,  ZF_3(4) ,  ZF_3(5)  ,  ZF_3(6)  ]
     %ZW_3  = [ ZW_3(1)  ,  ZW_3(2) ,  ZW_3(3)  ,  ZW_3(4) ,  ZW_3(5)  ,  ZW_3(6)  ]; Convect ZF_3 in the current coord system
    
    %Setup the matrix MVN for the nonpenetration condition
    %MVN=| MVNs_11, MVNs_12 |, where MVNs_ij are  (m x m) submatrices.
    %    | MVNs_21, MVNs_22 |
    %MVNs_11 and MVNs_22 are time independent; calculated once at the start and stored
    %MVNs_12 and MVNs_21 are time dependent; calculated at each time
    
    %%%CALCULATION OF MVNs_11 and MVNs_22
    %Use the wing-fixed coordinate system to calcuate the sub-matrix coefficients. 
    %The sub-matrix coefficients obtained using the global system are identical to
    %these and remain constant throughout the time steps.
    for i=1:nwing
    [MVNs(:,:,i)]=selfMatrixCoef( xv(i,:),yv(i,:),xc(i,:),yc(i,:),dfc(i,:),m); %Calcuate only once and keep using for eaxh step
    end
    
    if vfplot == 1
        switch vplotFR 
        case 1
            %Generate a cartesian mesh in the forward wing-fixed system for velocity plot
            if camber(1) == 0.0
            [ZETA] = igcMESH(c_(1),d_(1));
            else
            [ZETA] = igcamberMESH(c_(1),d_(1), camber(1));
            end
        case 2
            %Generate a cartesian mesh in the forward wing-fixed system for velocity plot
            if camber(2) == 0.0
            [ZETA] = igcMESH(c_(2),d_(2));
            else
            [ZETA] = igcamberMESH(c_(2),d_(2), camber(2));
            end
        end
    end
    %Start time marching
    for istep =1:nstep
        t=(istep-1)*dt; %Use (istep-1) to start with time=0
        %Get airfoil motion parameters=====================================
        for i=1:nwing          
            [alp(i),l(i),h(i),dalp(i), dl(i),dh(i)] = dfairfoilM(mpath(i),t,rt(i),tau(i),d(i),e(i),b(i),beta(i),delta,gMax(i),p(i),rtOff(i),U,V);
            LDOT(i,istep)=dl(i);
            HDOT(i,istep)=dh(i);
        end
 
        %Wing votex(V)/collocation(C) points & wake(W) vortex points=======
        % ZV(i,:), ZC(i,:), ZW(i,:)    (global system)
        % ZVb(i,:),ZCb(i,:),ZWb(i,:)   (body translating system)
           % ZW in istep=1 is assigned zero (or null) by initialization
           % ZW=ZF for istep >=2 (see the last command of the time marching loop)       
        %Unit normal of the wing at the collocation points (global): NC(i,:) 
        
        %Body translating system : This is the default system for the calculation of force and moment
        if istep == 1    
            for i=1:nwing
            [NC(i,:),ZV(i,:),ZC(i,:),ZVb(i,:),ZCb(i,:),ZWb(i,1:2)]...
                = dfwing2global(istep,t,a(i),alp(i),l(i),h(i),xv(i,:),yv(i,:),xc(i,:),yc(i,:),dfc(i,:),ZW(i,:),U,V);
            end
        else
            for i=1:nwing
            [NC(i,:),ZV(i,:),ZC(i,:),ZVb(i,:),ZCb(i,:),ZWb(i,1:2*(istep-1))]...
                = dfwing2global(istep,t,a(i),alp(i),l(i),h(i),xv(i,:),yv(i,:),xc(i,:),yc(i,:),dfc(i,:),ZW(i,:),U,V);
            end   
        end
        
        %wing translational systems for force moment calculation
            % ZVF(i,:),ZCF(i,:),ZWF(i,;)   (forward wing translating system)
            % ZVR(i,:),ZCR(i,:),ZWR(i,:)   (rear    wing translating system)
            % ZCF(1,:) is needed for the forward wing velocity calculation at the collocation points
            % ZCR(2,:) is needed for the rear    wing velocity calculation at the collocation points
            if istep == 1
                %Forward wing translating system
                for i=1:nwing
                [ZVF(i,:),ZCF(i,:),ZWF(i,1:2)]...
                = dfwing2FRT(istep,a(i),alp(i),l(i),h(i),xv(i,:),yv(i,:),xc(i,:),yc(i,:),ZW(i,:),l(1),h(1));
                end
                %Rear wing translating system
                for i=1:nwing
                [ZVR(i,:),ZCR(i,:),ZWR(i,1:2)]...
                = dfwing2FRT(istep,a(i),alp(i),l(i),h(i),xv(i,:),yv(i,:),xc(i,:),yc(i,:),ZW(i,:),l(2),h(2));
                end 
            else
                %Forward wing translating system
                for i=1:nwing
                [ZVF(i,:),ZCF(i,:),ZWF(i,1:2*(istep-1))]...
                = dfwing2FRT(istep,a(i),alp(i),l(i),h(i),xv(i,:),yv(i,:),xc(i,:),yc(i,:),ZW(i,:),l(1),h(1));
                end
                %Rear wing translating system
                for i=1:nwing
                [ZVR(i,:),ZCR(i,:),ZWR(i,1:2*(istep-1))]...
                = dfwing2FRT(istep,a(i),alp(i),l(i),h(i),xv(i,:),yv(i,:),xc(i,:),yc(i,:),ZW(i,:),l(2),h(2));
                end 
            end
        
        %Normal Velocity on the airfoil due to its motion==================
            %for iwing=1
            [VN(1,:)] = dfairfoilV(1,ZC(1,:),ZCF(1,:),NC(1,:),t,dl(1),dh(1),dalp(1));
            %for iwing=2
            [VN(2,:)] = dfairfoilV(2,ZC(2,:),ZCR(2,:),NC(2,:),t,dl(2),dh(2),dalp(2));  
        
 %%%%%%%%%%iGAMAw=2*(istep-1)%%%%%%%%%%       
        %Normal velocity on the airfoil due to the wake vortex=============
        for iwing=1:nwing
            [VNW(i,:)] = dfnVelocityw(m,ZC(i,:),NC(i,:),nwing,ZF,GAMAw,iGAMAw); %produces zero VNW for step 1 %ZF=ZW
        end
              
        %Calculation of time dependent sub-matrices MVN_12 and MVN_21======
        [MVNs_12]=crossMatrixCoef( ZV(2,:),ZC(1,:),dfc(1,:),m);
        [MVNs_21]=crossMatrixCoef( ZV(1,:),ZC(2,:),dfc(2,:),m);
        
        %Assemble MVNs(:,:,2), MVNs_12(:,:),MVNs_21(:,:) to a total matrix=
        [ MVN ] = assembleMatrix( m,MVNs,MVNs_12,MVNs_21 );
        
        %Solve the system of equations=====================================
        [ GAMA ] = dfsolution(m,MVN,VN,VNW,sGAMAw); 
        %Split GAMA into two parts
        GAM(1,1:m)=GAMA   (1 :  m); %Forward wing
        GAM(2,1:m)=GAMA((m+1):2*m); %Rear wing
        
        
        
        %Plot locations, ZW, of the wake vortices for the current step
        %in the space-fixed system
        dfplotVortexw(istep,nwing,iGAMAw,ZV,ZW);
        
        %Calculate the linear and angular impiulses on the airfoil=========
        %Include all of the bound vortices and wake vortices.
        %For istep=1, there is no wake vortices.
        
        %Use the body-translating system 
        [Limpb,Aimpb,Limpw,Aimpw] = dfimpulses(ZVb,ZWb, GAM,m,GAMAw,iGAMAw);
        Limpulseb(:,istep)=Limpb(:); Aimpulseb(:,istep)=Aimpb(:);
        Limpulsew(:,istep)=Limpw(:); Aimpulsew(:,istep)=Aimpw(:);      
        %Use the fore wing-translating system 
        [LimpbF,AimpbF,LimpwF,AimpwF] = dfimpulses(ZVF,ZWF, GAM,m,GAMAw,iGAMAw); 
        LimpulsebF(:,istep)=LimpbF(:); AimpulsebF(:,istep)=AimpbF(:);
        LimpulsewF(:,istep)=LimpwF(:); AimpulsewF(:,istep)=AimpwF(:); 
        %Use the rear wing-translating system 
        [LimpbR,AimpbR,LimpwR,AimpwR] = dfimpulses(ZVR,ZWR, GAM,m,GAMAw,iGAMAw);
        LimpulsebR(:,istep)=LimpbR(:); AimpulsebR(:,istep)=AimpbR(:);
        LimpulsewR(:,istep)=LimpwR(:); AimpulsewR(:,istep)=AimpwR(:);
        if vfplot ==1
        %Plot velocity field
            switch vplotFR
            case 1 %velocity around forard wing
                dfplotVelocity(istep,ZV,ZW,a(1),GAM,m,GAMAw,iGAMAw,alp(1),l(1),h(1));
            case 2 %velocity around ear wing
                dfplotVelocity(istep,ZV,ZW,a(2),GAM,m,GAMAw,iGAMAw,alp(2),l(2),h(2));
            end
        end
     
 %%%%%%%%%%iGAMAf=2*istep%%%%%%%%%%
         
        %Calculate the velocity at the free and wake vortices to be shed or convected
        for i=1:nwing
            iGAMAf(i)=2*istep;
            ZF(i,2*istep-1)=ZV(i,1); %append the coordinate of the leading edge
            ZF(i,2*istep  )=ZV(i,m); %append the coordinate of the trainling 
        end
        [VELF]=dfvelocity(ZF,iGAMAf,GAM,m,ZV,GAMAw,iGAMAw);
        
        %Convect GAMAf from ZF to ZW
        [ZW]=dfconvect(ZF,VELF,dt,iGAMAf);
        
        %Increment the number of wake vortices
        for i=1:nwing
            iGAMAw(i)=iGAMAw(i)+2;
            GAMAw(i,2*istep-1)=GAM(i,1);
            GAMAw(i,2*istep  )=GAM(i,m);
            %Add the new wake vortices from the current step
            sGAMAw(i)=sGAMAw(i)+GAM(i,1)+GAM(i,m);
        end
        
%%%%%%%%%%iGAMAw=2*istep%%%%%%%%%%          
    
        %All the convected vortices become wake vortices
        %Set these wake vortex to be the free vortices in the next step
        %,where two more free vortices (leading and trailing edge vortices)
        %will be added before convection
        ZF=ZW;
    end
   
    %Calculate the dimensional force and moment on the airfoil
    dfforceMoment(rho_,v_,d_(1), nstep,dt,U,V);
    
    %Print and plot the magnitudes of the dimensional wake vortex
    dfplotMVortexw(v_,d_(1),GAMAw,nstep);
    
otherwise
    
end %end of switch
   
%close opened files
    fclose('all');

end

