clear;close all;clc
 
t=linspace(0,2*pi,1000);
count=1;
h = figure(1);
set(h,'Position',[100 678 560 420])
 
for freqrps=.1:.1:2*pi
          y=sin(freqrps*t);
          plot(t,y);
          xlabel('x');
          ylabel('y');
          axis([0,2*pi,-1,1])
          S1=sprintf('y(t)=sin(%.2f t)',freqrps);
          text(2,.6,S1)
          freqcps=freqrps/(2*pi);
          S2=sprintf('frequency=%.2f rads/sec (%.2f cyc/sec)',freqrps,freqcps);
          text(2,.4,S2)
          title('Sinusoidal Function');
          M(count)=getframe(h);
          count=count+1;
end
 
movie2avi(M, 'sinusoid.avi', 'compression', 'none');