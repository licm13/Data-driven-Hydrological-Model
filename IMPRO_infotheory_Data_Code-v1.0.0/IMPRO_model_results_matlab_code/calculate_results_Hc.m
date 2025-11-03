% Uwe Ehret, 2024/01/26
% Calculates and plots results (Hc) from all models
clear all
close all
clc

%% settings
% - DateTime 5844 time steps [d]. This includes
%   - warmup for calibration (1.1.2000-31.12.2000)  1:366     (366)
%   - calibration (1.1.2001-31.12.2010)             367:4018  (3652)
%   - warmup for validation (1.1.2011-31.12.2011)   4019:4383 (365)
%   - validation (1.1.2012-31.12.2015)              4384:5844 (1461)

% % Iller
% basinstr = "iller";   % label of basin
% gaugestr = "Q_wibl";  % target gauge name
% edges_Q = cell(1,1);  % edges for Qobs and Qsim (see "this_and_that")
% edges_Q{1} = [0 23 35 55 80 100 128 180 230 270 395 610 700]; 

% % Saale
% basinstr = "saale";   % label of basin
% gaugestr = "Q_blank"; % target gauge name (iller Q_wibl  saale Q_blank  selke Q_haus
% edges_Q = cell(1,1);  % edges for Qobs and Qsim (see "this_and_that")
% edges_Q{1} = [0 7 9 15 20 30 35 50 60 85 110 150 185]; 

% Selke
basinstr = "selke";   % label of basin
gaugestr = "Q_haus";  % target gauge name
edges_Q = cell(1,1);  % edges for Qobs and Qsim (see "this_and_that")
edges_Q{1} = [0 0.5 1 2 3 5 7 9 12 18 25 40 45.5]; 

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
Qobs_binned = f_binme(Qobs,edges_Q); % bin the observed data

%% Climatology statistics
% uses mean of the observed time series as prediction

modelstr = "MEAN"; 
Hc = NaN(nss,1);
Qsim = NaN(size(Qobs));
Qsim(:) = mean(Qobs);
Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
Hc(:) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))

%% HBV statistics

% load data
load CONCEPTUAL_MODELS_results.mat *HBV*

modelstr = "HBV"; 
Hc = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 

  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
    Hc(s,r) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
  end

end

% % calculate Hc median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% Hc_median = median(Hc,2,"omitmissing");
% Hc_prctiles = prctile(Hc,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))
% eval(strcat("Hc_median_",basinstr,"_", modelstr, " = Hc_median;"))
% eval(strcat("Hc_prctiles_",basinstr,"_", modelstr, " = Hc_prctiles;"))

%% mHM statistics

% load data
load CONCEPTUAL_MODELS_results.mat *mHM*

modelstr = "mHM"; 

Hc = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
    Hc(s,r) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
  end

end

% % calculate Hc median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% Hc_median = median(Hc,2,"omitmissing");
% Hc_prctiles = prctile(Hc,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))
% eval(strcat("Hc_median_",basinstr,"_", modelstr, " = Hc_median;"))
% eval(strcat("Hc_prctiles_",basinstr,"_", modelstr, " = Hc_prctiles;"))

%% Raven GR4J statistics

% load data
load CONCEPTUAL_MODELS_results.mat *Raven*
modelstr = "Raven_GRJ4"; 
Hc = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
    Hc(s,r) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
  end

end

% % calculate Hc median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% Hc_median = median(Hc,2,"omitmissing");
% Hc_prctiles = prctile(Hc,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))
% eval(strcat("Hc_median_",basinstr,"_", modelstr, " = Hc_median;"))
% eval(strcat("Hc_prctiles_",basinstr,"_", modelstr, " = Hc_prctiles;"))

%% SWAT statistics

% load data
load CONCEPTUAL_MODELS_results.mat *SWAT*
modelstr = "SWAT"; 
Hc = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
    Hc(s,r) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
  end

end

% % calculate Hc median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% Hc_median = median(Hc,2,"omitmissing");
% Hc_prctiles = prctile(Hc,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))
% eval(strcat("Hc_median_",basinstr,"_", modelstr, " = Hc_median;"))
% eval(strcat("Hc_prctiles_",basinstr,"_", modelstr, " = Hc_prctiles;"))

%% HBV random statistics

% load data
load CONCEPTUAL_MODELS_results.mat *HBV_random*
modelstr = "HBV_random"; 
Hc = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
    Hc(s,r) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
  end

end

% % calculate Hc median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% Hc_median = median(Hc,2,"omitmissing");
% Hc_prctiles = prctile(Hc,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))
% eval(strcat("Hc_median_",basinstr,"_", modelstr, " = Hc_median;"))
% eval(strcat("Hc_prctiles_",basinstr,"_", modelstr, " = Hc_prctiles;"))

%% HBV lumped statistics

% load data
load CONCEPTUAL_MODELS_results.mat *HBV_lumped*
modelstr = "HBV_lumped"; 
Hc = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
    Hc(s,r) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
  end

end

% % calculate Hc median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% Hc_median = median(Hc,2,"omitmissing");
% Hc_prctiles = prctile(Hc,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))
% eval(strcat("Hc_median_",basinstr,"_", modelstr, " = Hc_median;"))
% eval(strcat("Hc_prctiles_",basinstr,"_", modelstr, " = Hc_prctiles;"))

%% HBV dp statistics
% Note: 
% - only 1 sampling repetition!
% - therefore only Hc is stored, not Hc_median and no Hc_prctiles

% load data
load CONCEPTUAL_MODELS_results.mat *HBV_dp*
modelstr = "HBV_dp"; 
Hc = NaN(nss,1);

% loop over all sample sizes
for s = 1 : nss 
  evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,1);");
  eval(evalstr);
  Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
  Hc(s,1) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
end

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))


%% EDDIS statistics

% load data
load EDDIS_results.mat *Ps1t2Ts1t3_numbin_2*
modelstr = "EDDIS_Ps1t2Ts1t3_numbin_2"; % Ps1t2Ts1t3 with 2 bins each was the best model for all basins
Hc = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
    Hc(s,r) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
  end

end

% % calculate Hc median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% Hc_median = median(Hc,2,"omitmissing");
% Hc_prctiles = prctile(Hc,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))
% eval(strcat("Hc_median_",basinstr,"_", modelstr, " = Hc_median;"))
% eval(strcat("Hc_prctiles_",basinstr,"_", modelstr, " = Hc_prctiles;"))

%% RTREE statistics

% load data
load RTREE_results.mat *Ps1t2Ts1t3*
modelstr = "RTREE_Ps1t2Ts1t3"; % Ps1t2Ts1t3 was the best model for all basins
Hc = NaN(nss,nrep);

% loop over all sample sizes
for s = 1 : nss 
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
    Hc(s,r) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
  end

end

% % calculate Hc median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% Hc_median = median(Hc,2,"omitmissing");
% Hc_prctiles = prctile(Hc,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))
% eval(strcat("Hc_median_",basinstr,"_", modelstr, " = Hc_median;"))
% eval(strcat("Hc_prctiles_",basinstr,"_", modelstr, " = Hc_prctiles;"))

%% LSTM statistics

% load data
load LSTM_ANN_results.mat
modelstr = "LSTM"; 
Hc = NaN(nss,nrep);

% loop over all sample sizes
for s = 2 : nss % NOTE: s=2 because LSTM has no sample size 10!
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
    Hc(s,r) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
  end

end

% % calculate Hc median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% Hc_median = median(Hc,2,"omitmissing");
% Hc_prctiles = prctile(Hc,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))
% eval(strcat("Hc_median_",basinstr,"_", modelstr, " = Hc_median;"))
% eval(strcat("Hc_prctiles_",basinstr,"_", modelstr, " = Hc_prctiles;"))

%% ANN statistics

% load data
load LSTM_ANN_results.mat
modelstr = "ANN"; 
Hc = NaN(nss,nrep);

% loop over all sample sizes
for s = 2 : nss % NOTE: s=2 because LSTM has no sample size 10!
  
  % loop over all sampling repetitions
  for r = 1 : nrep
    evalstr = strcat("Qsim = ",basinstr, "_",modelstr,"_samplesize",samplesizestr(s),"(:,r);");
    eval(evalstr);
    Qsim_binned = f_binme(Qsim,edges_Q,true); % bin the simulated data
    Hc(s,r) = f_conditionalentropy(Qobs_binned,Qsim_binned); % calculate Hc
  end

end

% % calculate Hc median and percentiles
% % percentiles are calculated from sorted values and (if necessary) linear interpolation
% Hc_median = median(Hc,2,"omitmissing");
% Hc_prctiles = prctile(Hc,prctiles,2); % [lower, upper] percentile

% rename results
eval(strcat("Hc_",basinstr,"_", modelstr, " = Hc;"))
% eval(strcat("Hc_median_",basinstr,"_", modelstr, " = Hc_median;"))
% eval(strcat("Hc_prctiles_",basinstr,"_", modelstr, " = Hc_prctiles;"))

% %% plot Hc
% 
% figure
% hold on
% 
% % Climatology
% modelstr = "MEAN";
% colorstr = "black";
% eval(strcat("plot(sample_sizes,Hc_",basinstr,"_",modelstr,",'Color',rgb('",colorstr,"'),'DisplayName','",modelstr,"');"));
% 
% % HBV
% modelstr = "HBV";
% colorstr = "darkviolet";
% evalstr = strcat("bounds = [abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,Hc_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % mHM
% modelstr = "mHM";
% colorstr = "steelblue";
% evalstr = strcat("bounds = [abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,Hc_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % Raven GRJ4
% modelstr = "Raven_GRJ4";
% colorstr = "tomato";
% evalstr = strcat("bounds = [abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,Hc_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % SWAT
% modelstr = "SWAT";
% colorstr = "seagreen";
% evalstr = strcat("bounds = [abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,Hc_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % EDDIS
% modelstr = "EDDIS_Ps1t2Ts1t3_numbin_2";
% colorstr = "darkkhaki";
% evalstr = strcat("bounds = [abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,Hc_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = "EDDIS";
% hl.Color = rgb(colorstr);
% hp.DisplayName = "EDDIS";
% hp.FaceColor = rgb(colorstr);
% 
% % RTREE
% modelstr = "RTREE_Ps1t2Ts1t3";
% colorstr = "hotpink";
% evalstr = strcat("bounds = [abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,Hc_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = "RTREE";
% hl.Color = rgb(colorstr);
% hp.DisplayName = "RTREE";
% hp.FaceColor = rgb(colorstr);
% 
% % LSTM
% modelstr = "LSTM";
% colorstr = "orange";
% evalstr = strcat("bounds = [abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,Hc_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
%   
% % ANN
% modelstr = "ANN";
% colorstr = "cyan";
% evalstr = strcat("bounds = [abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,Hc_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % HBV random
% modelstr = "HBV_random";
% colorstr = "magenta";
% evalstr = strcat("bounds = [abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,1)) abs(Hc_median_",basinstr,"_",modelstr," - Hc_prctiles_",basinstr,"_",modelstr,"(:,2))];");
% eval(evalstr);
% evalstr = strcat("[hl,hp] = boundedline(sample_sizes,Hc_median_",basinstr,"_",modelstr,",bounds,'alpha','transparency', 0.1);");
% eval(evalstr);
% hl.DisplayName = modelstr;
% hl.Color = rgb(colorstr);
% hp.DisplayName = modelstr;
% hp.FaceColor = rgb(colorstr);
% 
% % HBV dp
% modelstr = "HBV_dp";
% colorstr = "thistle";
% eval(strcat("plot(sample_sizes,Hc_",basinstr,"_",modelstr,",'Color',rgb('",colorstr,"'),'DisplayName','",modelstr,"');"));
% 
% % HBV dp dds
% modelstr = "HBV_dp_dds";
% colorstr = "thistle";
% eval(strcat("plot(sample_sizes,Hc_",basinstr,"_",modelstr,",'Color',rgb('",colorstr,"'),'DisplayName','",modelstr,"');"));
% 
% hold off
% title(strcat("Hc ",basinstr))
% xticks(sample_sizes)
% %ylim([0,log2(12)])
% % yticks(-1:0.1:1)
% xlabel("training sample size")
% ylabel("testing performance")
% legend


