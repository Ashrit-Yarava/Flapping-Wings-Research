%Sinusoidal Animation
 
%       Let?s plot a sinusoid with the frequency changing before our eyes.  Copy the following code into an m-file (sinmovie.m).
 
% Script file (sinmovie.m) to plot an animation of a sin function with increasing
% frequency.
%                                 
% Author?
% Date?
 
clear;close all;clc
 
t=linspace(0,2*pi,1000);
count=1;
figure
 
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
          M(count)=getframe;
          count=count+1;
end
 
movie(M,2,10);
