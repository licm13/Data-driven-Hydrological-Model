function [kge] = f_KGE(obs, sim)
% Returns Kling-Gupta efficiency
% Best possible score is 1, bigger value is better. Range = [-inf, 1]
% Corresponding paper:
% Gupta, H. V., H. Kling, K. K. Yilmaz, and G. F. Martinez (2009), Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling, Journal of Hydrology, 377(1), 80-91.

cc = corr(obs,sim);
alpha = std(sim) / std(obs);
beta = sum(sim) / sum(obs);
kge = 1 - sqrt((cc - 1)^2 + (alpha - 1)^2 + (beta - 1)^2);
        
end