clc;
clear;
close all;

constant = -0.5;
params = [0.5, 0.3, 0.4, -0.4, 0.1^2];
arma_part = [2, 2];
n_cycle = 1000;
y = ARMA_sample(constant, params, arma_part, n_cycle);
plot(y, 'linewidth', 2);
xlabel('Time step');
ylabel('Load');