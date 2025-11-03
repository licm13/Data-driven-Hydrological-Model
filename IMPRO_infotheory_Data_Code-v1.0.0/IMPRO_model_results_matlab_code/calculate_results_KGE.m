% Uwe Ehret, 2024/01/26
% Calculates and plots results from all models
clear all
close all
clc

%% settings
% - DateTime 5844 time steps [d]. This includes
%   - warmup for calibration (1.1.2000-31.12.2000)  1:366     (366)
%   - calibration (1.1.2001-31.12.2010)             367:4018  (3652)
%   - warmup for validation (1.1.2011-31.12.2011)   4019:4383 (365)
%   - validation (1.1.2012-31.12.2015)              4384:5844 (1461)

% Iller
basinstr = "iller";   % label of basin
gaugestr = "Q_wibl";  % target gauge name

% % Saale
% basinstr = "saale";   % label of basin
% gaugestr = "Q_blank"; % target gauge name (iller Q_wibl  saale Q_blank  selke Q_haus

% % Selke
% basinstr = "selke";   % label of basin
% gaugestr = "Q_haus";  % target gauge name

prctiles = [25 75]; % percentiles for which bounds should be plotted
samplesizestr = ["10","50", "100", "250", "500", "1000", "2000", "3000", "3652"];
sample_sizes = [10 50 100 250 500 1000 2000 3000 3652];     % sample sizes
nss = length(sample_sizes);                            % number of sample sizes
nrep = 30; % number of sampling repetitions

%% load observed data

eval(strcat("load ", basinstr,"_data.mat DateTime ", gaugestr)); % load observed target data
eval(strcat("Qobs=", gaugestr,";"));
eval(strcat("clear ", gaugestr));
Qobs = Qobs(4384:end); % extract data from validation period
DateTimeobs = DateTime(4384:end); % extract data from validation period

%% Climatology statistics
% uses mean of the observed time series as prediction

modelstr = "MEAN"; 
KGE = NaN(nss,1);
Qsim = NaN(size(Qobs));
Qsim(:) = mean(Qobs);
KGE(:) = f_KGE(Qobs,Qsim); % calculate KGE

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr, " = KGE;"))

%% HBV statistics

% load data
load CONCEPTUAL_MODELS_results.mat *HBV*

modelstr = "HBV"; 
KGE = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    KGE(s,r) = f_KGE(Qobs,Qsim); % calculate KGE
  end

end

% % calculate KGE median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% KGE_median = median(KGE,2,"omitmissing");
% KGE_prctiles = prctile(KGE,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr, " = KGE;"))
% eval(strcat("KGE_median_",basinstr,"_", modelstr, " = KGE_median;"))
% eval(strcat("KGE_prctiles_",basinstr,"_", modelstr, " = KGE_prctiles;"))

%% mHM statistics

% load data
load CONCEPTUAL_MODELS_results.mat *mHM*

modelstr = "mHM"; 

KGE = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    KGE(s,r) = f_KGE(Qobs,Qsim); % calculate KGE
  end

end

% % calculate KGE median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% KGE_median = median(KGE,2,"omitmissing");
% KGE_prctiles = prctile(KGE,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr, " = KGE;"))
% eval(strcat("KGE_median_",basinstr,"_", modelstr, " = KGE_median;"))
% eval(strcat("KGE_prctiles_",basinstr,"_", modelstr, " = KGE_prctiles;"))

%% Raven GRJ4 statistics

% load data
load CONCEPTUAL_MODELS_results.mat *Raven*
modelstr = "Raven_GRJ4"; 
KGE = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    KGE(s,r) = f_KGE(Qobs,Qsim); % calculate KGE
  end

end

% % calculate KGE median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% KGE_median = median(KGE,2,"omitmissing");
% KGE_prctiles = prctile(KGE,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr, " = KGE;"))
% eval(strcat("KGE_median_",basinstr,"_", modelstr, " = KGE_median;"))
% eval(strcat("KGE_prctiles_",basinstr,"_", modelstr, " = KGE_prctiles;"))

%% SWAT statistics

% load data
load CONCEPTUAL_MODELS_results.mat *SWAT*
modelstr = "SWAT"; 
KGE = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep-2 % NOTE: nss-2 because SWAT only has 28 repetitions!
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    KGE(s,r) = f_KGE(Qobs,Qsim); % calculate KGE
  end

end

% % calculate KGE median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% KGE_median = median(KGE,2,"omitmissing");
% KGE_prctiles = prctile(KGE,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr, " = KGE;"))
% eval(strcat("KGE_median_",basinstr,"_", modelstr, " = KGE_median;"))
% eval(strcat("KGE_prctiles_",basinstr,"_", modelstr, " = KGE_prctiles;"))

%% HBV random statistics

% load data
load CONCEPTUAL_MODELS_results.mat *HBV_random*
modelstr = "HBV_random"; 
KGE = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 

  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    KGE(s,r) = f_KGE(Qobs,Qsim); % calculate KGE
  end

end

% % calculate KGE median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% KGE_median = median(KGE,2,"omitmissing");
% KGE_prctiles = prctile(KGE,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr, " = KGE;"))
% eval(strcat("KGE_median_",basinstr,"_", modelstr, " = KGE_median;"))
% eval(strcat("KGE_prctiles_",basinstr,"_", modelstr, " = KGE_prctiles;"))

%% HBV dp statistics
% Note: 
% - only 1 sampling repetition!
% - therefore only KGE is stored, not KGE_median and no KGE_prctiles

% load data
load CONCEPTUAL_MODELS_results.mat *HBV_dp*
modelstr = "HBV_dp"; 
KGE = NaN(nss,1);

% loop over all sample sizes
for s = 1 : nss 
  evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,1);");
  eval(evalstr);
  KGE(s,1) = f_KGE(Qobs,Qsim); % calculate KGE
end

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr, " = KGE;"))


%% HBV lumped statistics

% load data
load CONCEPTUAL_MODELS_results.mat *HBV*

modelstr = "HBV_lumped"; 
KGE = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    KGE(s,r) = f_KGE(Qobs,Qsim); % calculate KGE
  end

end

% % calculate KGE median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% KGE_median = median(KGE,2,"omitmissing");
% KGE_prctiles = prctile(KGE,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr, " = KGE;"))
% eval(strcat("KGE_median_",basinstr,"_", modelstr, " = KGE_median;"))
% eval(strcat("KGE_prctiles_",basinstr,"_", modelstr, " = KGE_prctiles;"))


%% EDDIS statistics

% load data
load EDDIS_results.mat *Ps1t2Ts1t3_numbin_2*
modelstr = "EDDIS_Ps1t2Ts1t3_numbin_2"; % Ps1t2Ts1t3 with 2 bins each was the best model for all basins
KGE = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    KGE(s,r) = f_KGE(Qobs,Qsim); % calculate KGE
  end

end

% % calculate KGE median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% KGE_median = median(KGE,2,"omitmissing");
% KGE_prctiles = prctile(KGE,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr," = KGE;"))
% eval(strcat("KGE_median_",basinstr,"_", modelstr," = KGE_median;"))
% eval(strcat("KGE_prctiles_",basinstr,"_", modelstr," = KGE_prctiles;"))

%% RTREE statistics

% load data
load RTREE_results.mat *Ps1t2Ts1t3*
modelstr = "RTREE_Ps1t2Ts1t3"; % Ps1t2Ts1t3 was the best model for all basins
KGE = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    KGE(s,r) = f_KGE(Qobs,Qsim); % calculate KGE
  end

end

% % calculate KGE median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% KGE_median = median(KGE,2,"omitmissing");
% KGE_prctiles = prctile(KGE,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr," = KGE;"))
% eval(strcat("KGE_median_",basinstr,"_", modelstr," = KGE_median;"))
% eval(strcat("KGE_prctiles_",basinstr,"_", modelstr," = KGE_prctiles;"))

%% LSTM statistics

% load data
load LSTM_ANN_results.mat
modelstr = "LSTM"; 
KGE = NaN(nss,nrep);

% loop over all sample sizes
for s = 2 : nss % NOTE: s=2 because LSTM has no sample size 10!

  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    KGE(s,r) = f_KGE(Qobs,Qsim); % calculate KGE
  end

end

% % calculate KGE median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% KGE_median = median(KGE,2,"omitmissing");
% KGE_prctiles = prctile(KGE,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr, " = KGE;"))
% eval(strcat("KGE_median_",basinstr,"_", modelstr, " = KGE_median;"))
% eval(strcat("KGE_prctiles_",basinstr,"_", modelstr, " = KGE_prctiles;"))

%% ANN statistics

% load data
load LSTM_ANN_results.mat
modelstr = "ANN"; 
KGE = NaN(nss,nrep);

% loop over all sample sizes
for s = 2 : nss % NOTE: s=2 because LSTM has no sample size 10!

  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    KGE(s,r) = f_KGE(Qobs,Qsim); % calculate KGE
  end

end

% % calculate KGE median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% KGE_median = median(KGE,2,"omitmissing");
% KGE_prctiles = prctile(KGE,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("KGE_",basinstr,"_", modelstr, " = KGE;"))
% eval(strcat("KGE_median_",basinstr,"_", modelstr, " = KGE_median;"))
% eval(strcat("KGE_prctiles_",basinstr,"_", modelstr, " = KGE_prctiles;"))

% %% plot time series
% % always pick the first repetition of sample size 3652
% 
% figure
% hold on
% % OBSERVED
% plot(DateTimeobs,Qobs,'DisplayName','Observed','Color',rgb('black'),'LineWidth',2);
% % HBV
% eval(strcat("plot(DateTimeobs,",basinstr,"_HBV_samplesize3652(:,1),'DisplayName','HBV','Color',rgb('darkviolet'));"))
% % mHM
% eval(strcat("plot(DateTimeobs,",basinstr,"_mHM_samplesize3652(:,1),'DisplayName','mHM','Color',rgb('steelblue'));"))
% % Raven_blended
% eval(strcat("plot(DateTimeobs,",basinstr,"_Raven_blended_samplesize3652(:,1),'DisplayName','Raven blended','Color',rgb('tomato'));"))
% % Raven_GRJ4
% eval(strcat("plot(DateTimeobs,",basinstr,"_Raven_blended_samplesize3652(:,1),'DisplayName','Raven GRJ4','Color',rgb('tomato'));"))
% % SWAT
% eval(strcat("plot(DateTimeobs,",basinstr,"_SWAT_samplesize3652(:,1),'DisplayName','SWAT','Color',rgb('seagreen'));"))
% % EDDIS
% eval(strcat("plot(DateTimeobs,",basinstr,"_EDDIS_Ps1t2Ts1t3_numbin_2_samplesize3652(:,1),'DisplayName','EDDIS','Color',rgb('darkkhaki'));"))
% % RTREE
% eval(strcat("plot(DateTimeobs,",basinstr,"_RTREE_Ps1t2Ts1t3_samplesize3652(:,1),'DisplayName','RTREE','Color',rgb('hotpink'));"))
% % LSTM
% eval(strcat("plot(DateTimeobs,",basinstr,"_LSTM_samplesize3652(:,1),'DisplayName','LSTM','Color',rgb('orange'));"))
% % ANN
% eval(strcat("plot(DateTimeobs,",basinstr,"_ANN_samplesize3652(:,1),'DisplayName','ANN','Color',rgb('cyan'));"))
% % HBV random
% eval(strcat("plot(DateTimeobs,",basinstr,"_HBV_random_samplesize3652(:,1),'DisplayName','HBV random','Color',rgb('magenta'));"))
% % HBV dp
% eval(strcat("plot(DateTimeobs,",basinstr,"_HBV_dp_samplesize3652(:,1),'DisplayName','HBV dp','Color',rgb('thistle'));"))
% % HBV dp dds
% eval(strcat("plot(DateTimeobs,",basinstr,"_HBV_dp_dds_samplesize3652(:,1),'DisplayName','HBV dp dds','Color',rgb('thistle'));"))
% 
% hold off
% title(basinstr)
% ylabel("streamflow [m3/s]")
% legend

% %% plot KGE
% 
% figure
% hold on
% 
% % Climatology
% modelstr = "MEAN";
% colorstr = "black";
% eval(strcat("plot(sample_sizes,KGE_",basinstr,"_",modelstr,",'Color',rgb('",colorstr,"'),'DisplayName','",modelstr,"');"));
% 
% % HBV
% modelstr = "HBV";
% colorstr = "darkviolet";
% evalstr = strcat("bounds = [abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,KGE_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % mHM
% modelstr = "mHM";
% colorstr = "steelblue";
% evalstr = strcat("bounds = [abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,KGE_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % Raven blended
% modelstr = "Raven_blended";
% colorstr = "tomato";
% evalstr = strcat("bounds = [abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,KGE_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % Raven GRJ4
% modelstr = "Raven_GRJ4";
% colorstr = "tomato";
% evalstr = strcat("bounds = [abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,KGE_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % SWAT
% modelstr = "SWAT";
% colorstr = "seagreen";
% evalstr = strcat("bounds = [abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,KGE_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % EDDIS
% modelstr = "EDDIS_Ps1t2Ts1t3_numbin_2";
% colorstr = "darkkhaki";
% evalstr = strcat("bounds = [abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,KGE_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = "EDDIS";
% hl.Color = rgb(colorstr);
% hp.DisplayName = "EDDIS";
% hp.FaceColor = rgb(colorstr);
% 
% % RTREE
% modelstr = "RTREE_Ps1t2Ts1t3";
% colorstr = "hotpink";
% evalstr = strcat("bounds = [abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,KGE_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = "RTREE";
% hl.Color = rgb(colorstr);
% hp.DisplayName = "RTREE";
% hp.FaceColor = rgb(colorstr);
% 
% % LSTM
% modelstr = "LSTM";
% colorstr = "orange";
% evalstr = strcat("bounds = [abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,KGE_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
%   
% % ANN
% modelstr = "ANN";
% colorstr = "cyan";
% evalstr = strcat("bounds = [abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,KGE_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % HBV random
% modelstr = "HBV_random";
% colorstr = "magenta";
% evalstr = strcat("bounds = [abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(KGE_median_",basinstr,"_",modelstr," - KGE_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,KGE_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % HBV dp
% modelstr = "HBV_dp";
% colorstr = "thistle";
% eval(strcat("plot(sample_sizes,KGE_",basinstr,"_",modelstr,",'Color',rgb('",colorstr,"'),'DisplayName','",modelstr,"');"));
% 
% % HBV dp dds
% modelstr = "HBV_dp_dds";
% colorstr = "thistle";
% eval(strcat("plot(sample_sizes,KGE_",basinstr,"_",modelstr,",'Color',rgb('",colorstr,"'),'DisplayName','",modelstr,"');"));
% 
% hold off
% title(strcat("KGE ",basinstr))
% xticks(sample_sizes)
% % ylim([-1,1])
% % yticks(-1:0.1:1)
% xlabel("training sample size")
% ylabel("testing performance")
% legend


