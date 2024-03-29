
function [dr, drl, ai, g] = audio_pre_processing(filename, lambda)

%Read in wav file
[y, Fs] = audioread(filename);
dt = 1/Fs;
t = 0:dt:(length(y)*dt)-dt;
%Change sampling rate to 44100
[P,Q] = rat(44.1e3/Fs);
abs(P/Q*Fs-44100);
y = resample(y,P,Q);

%interpolate to uniform sampling rate 44,100Hz
% ti = (0:1/srate:t(end))';
% ai = interp1(t,a,ti,'cubic');
% t = ti;
% a = ai;

%segement original signal into short-duration 10-ms windows and extract
%energy of each window
N = length(y);
ts = 0.01; %Frame step in seconds 10-ms
frame_length = floor(ts*44100);
%y2 = buffer(y,frame_length);
trailingsamples = mod(N, frame_length);
sampleframes = reshape(y(1:end-trailingsamples), frame_length, []);
[m,n] = size(sampleframes);
x = n:1;
for i = 1:n
    x(i,1) = sum(abs(sampleframes(:,i)).^2);
end
ai = y;
dr = x;
drl = log10(dr);
g = 0;
% Nsamps = length(y);
% ham = hamming(Nsamps);
% windowed = y .* ham;
% ham_fft = fft(windowed);
% ham_fft = ham_fft(1:Nsamps/2);
% PowSpec = abs(fft(ham_fft)).^2;
% plot(PowSpec);
% dr = PowSpec;


% window = 0.001/t(end,1) * length(y);
% g=0;
% ai = y;

%drl = sampleframes;

 plot(y);
 xlabel('Seconds');
 ylabel('Amplitude');

end