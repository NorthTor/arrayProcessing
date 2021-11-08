%% ====================================================================
% Array and sensor signal processing mini project 2 example
%%======================================================================
clc;
close all;
clear; 

load('MeasurementforMiniProject.mat')

N = size(X,2);
L = size(X,1);
c = 3e8;
f0 = 7.5e9;
delta_f = 2e6;
lambda = c / f0;
Ts = tau(2)-tau(1);         %delay spacing

X_r_f = X_synthetic;     

signal_power = (X_r_f(:)'*X_r_f(:))/(numel(X_r_f));

%% Add noise
SNR = 50;

noise_power = 10^(-SNR/10)*signal_power;
Noise_r_f = sqrt(noise_power/2)*(randn(size(X_r_f)) + 1j*randn(size(X_r_f)));

Y_r_f = X_r_f + Noise_r_f;

%% Definition of array
N_row = 71;
N_column = 66; 
Lf = 101
L1 = 6;
L2 = 6;
idx_array = getSubarray(N_row, N_column, L1, L2, 1);

idx_tau = 1:length(tau); %Note: We can also truncate the delay range if desired
Nf = length(idx_tau);

%selection of small array from measurement
Y_r_f_array = Y_r_f(idx_array,idx_tau);
r_array = r(:, idx_array);

%% Spatial Smoothing
SL1 = 6; %size of sub-array - x-axis
SL2 = 6; %y-axis
SL3 = 6; %frequency

%[R_fb] = spatialSmoothing(Y_r_f_array, SL1, SL2, SL3, L1, L2);

%% Bartlett 

azimuth_step = pi/25;
theta_search = (0:azimuth_step:2*pi);
theta_degree = theta_search.*180/pi;
delay_step = 4e-8 / 50;
step = 50;

tau_search = linspace(abs(tau(1)), 35e-9+abs(tau(1)), step);
%tau_search = (-4e-8:delay_step:4e-8);

Y_r_f_columnwise = reshape(Y_r_f_array, [], 1);

Lf_array = (0:1:Lf-1);


idx_subarray = getSubarray(77, 66, 4, 4, 1);
r_subarray = r(:, idx_subarray);

[R_fb, R_fb_column] = spatialSmoothing(Y_r_f_array, 4, 4, 101, 4, 4);
[P_music] = MUSIC(R_fb_column, r_subarray, theta_search, tau_search, delta_f, lambda, Lf_array, 5);

    
limits = 20*log10(max(max(abs(P_music)))) + [-40 0];
figure; imagesc(theta_degree, tau_search*c+tau(1)*c, 20*log10(abs(P_music).'),limits); colorbar
hold on;
scatter((smc_param.AoA+[0 0 0 2*pi 0]')*180/pi, smc_param.distance, 100, 'x', 'r');
title('Bartlett Spectrum');

    
function [P_bartlett] = bartlett(Y_r_f, r_position, theta_search, tau_search, delta_f, lambda, Lf_array)
    P_bartlett = zeros(length(theta_search), length(tau_search));
    R_hat = (Y_r_f * Y_r_f');
    for i = 1:length(theta_search)
        verif = i
        a = exp(1j*2*pi/lambda * [cos(theta_search(i)); sin(theta_search(i))]'* r_position);
        for k = 1:length(tau_search)
            b = exp(-1j*2*pi*Lf_array*tau_search(k)*delta_f);
            u = kron(b,a);
            u = u.';
            coef = u' * R_hat * u ;
            P_bartlett(i,k) = coef ;
        end
        
    end

end

function [P_capon] = capon(Y_r_f, r_position, theta_search, tau_search, delta_f, lambda, Lf_array)
    P_capon = zeros(length(theta_search), length(tau_search));
    R_hat = (Y_r_f * Y_r_f');
    R_inverse = inv(R_hat);
    for i = 1:length(theta_search)
        verif = i
        a = exp(1j*2*pi/lambda * [cos(theta_search(i)); sin(theta_search(i))]'* r_position);
        for k = 1:length(tau_search)
            b = exp(-1j*2*pi*Lf_array*tau_search(k)*delta_f);
            u = kron(b,a);
            u = u.';
            coef = u' * R_inverse * u ;
            P_capon(i,k) = 1 / coef ;
        end
        
    end
end

function [P_music] = MUSIC(R_hat, r_position, theta_search, tau_search, delta_f, lambda, Lf_array, M)
    L = length(R_hat);
    P_music = zeros(length(theta_search), length(tau_search));
    [U,E] = eig(R_hat);
    U_noise = U(:,1:L-M);
    for i = 1:length(theta_search)
        verif = i
        a = exp(1j*2*pi/lambda * [cos(theta_search(i)); sin(theta_search(i))]'* r_position);
        for k = 1:length(tau_search)
            b = exp(-1j*2*pi*Lf_array*tau_search(k)*delta_f);
            u = kron(b,a);
            u = u.';
            coef = u' * U_noise * U_noise' * u ;
            P_music(i,k) = 1 / coef ;
        end
        
    end
end

function [R_fb, R_fb_column] = spatialSmoothing(Y_r_f, SL1, SL2, SL3, L1, L2)
P1 = L1-SL1+1; %Number of sub-arrays - x-axis
P2 = L2-SL2+1; %y-axis


for j = 1:P2 %Incrementing on the y-axis
    for i = 1:P1 %Incrementing on the x-axis
        for k = 1:SL2 %Selects SL1 entries, moves L1-SL1 elements, selects the next SL1 elements... SL2 times into a (SL1*SL2),1 matrix
            SM((k-1)*SL1+1:SL1*k,:) = Y_r_f((k-1)*L1+(j-1)*L2+i:(k-1)*L1+SL1+i-1+(j-1)*L2,:);
            SM_column = reshape(SM,[],1);
        end
        R_p(:,:,(j-1)*P2+i)=SM*SM'; %Calculates R_f for a total of P1*P2 indicies
        R_p_column(:,:,(j-1)*P2+i) = SM_column*SM_column';  
    end

end

R_f = sum(R_p,3)/P1*P2; %Summing over 3rd dimensions (the P1*P2 indicies) and divides by P1. The division factor may be wrong
R_f_column = sum(R_p_column,3)/P1*P2;

J_s = flipud(eye(size(R_f)));
J_s_column = flipud(eye(size(R_f_column)));

R_fb = 0.5*(R_f+J_s*conj(R_f)*J_s);
R_fb_column = 0.5*(R_f_column+J_s_column*conj(R_f_column)*J_s_column);

end
