
function [dr, drl, ai, g] = audio_pre_processing(filename, lambda)

[y, Fs] = audioread(filename);

dt = 1/Fs;
t = 0:dt:(length(y)*dt)-dt;
a = y;

srate = 44100;
%interpolate to uniform sampling rate 44,100Hz
ti = (0:1/srate:t(end))';
ai = interp1(t,a,ti,'cubic');
t = ti;
a = ai;
%segement original signal into short-duration 10-ms windows

g=0;
dr = y;
drl = log10(dr);

 plot(t,a);
 xlabel('Seconds');
ylabel('Amplitude');

end
