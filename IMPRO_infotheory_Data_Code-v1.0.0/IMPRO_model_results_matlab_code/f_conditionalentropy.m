function [cH] = f_conditionalentropy(targets_binned, predictors_binned)
% Returns the (joint) conditional entropy of an 1-to-any-dimensional discrete (binned) frequency distribution of target variable(s) 
% given prior knowledge of 1-to-any number of discrete (binned) predictor values
% Note: This is done very fast as only non-zero bin occupations are considered
% Input
% - targets_binned: [num_data, num_dim_targets] matrix, where num_data is the number of data tuples (= sample size),
%   and num_dim_targets is the number of dimensions of the target data set (= number of target variables)
% - predictors_binned: [num_data, num_dim_predictors] matrix, where num_data is the number of data tuples (= sample size),
%   and num_dim_predictors is the number of dimensions of the predictor data set (= number of predictor variables)
%   Note
%   - targets and predictors must have the same length (num_data)
%   - targets and predictors must be in the same order, data tuples are linked by the same row number
%   - targets and predictors must be NaN-free
%   - values in targets and predictors are positive integers, indicating the bin number into which the original data were classified
% Output
% - cH: [1,1] conditional entropy in [bit]
% Dependencies
% - f_entropy
% Version
% - 2022/03/11 Uwe Ehret: Renamed (old name was f_conditional_entropy_anyd_fast)
% - 2021/07/15 Uwe Ehret: initial version

% check if 'targets_binned' and 'predictors_binned' have the same length
    if size(targets_binned,1) ~= size(predictors_binned,1)
        error('targets and predictors do not have the same length')
    end

% check if 'targets_binned' and 'predictors_binned' are NaN-free
    if ~isempty(find(isnan(targets_binned)))
        error('targets_binned contains NaNs')
    end
    if ~isempty(find(isnan(predictors_binned)))
        error('predictors_binned contains NaNs')
    end

% find unique predictor tuples
    [C,ia,ic] = unique(predictors_binned,'rows','stable');
    % If [C,ia,ic] = unique(A,'rows','stable'), then
    % C: list of unique rows in A, in order of appearance in A. Unique rows rather than unique colum values is ensured by parameter 'rows'
    % ia: C = A(ia), i.e. ia is the index, where in A each row in C occurs the first time (ensured by parameter 'stable') --> ia has the same length as C
    % ic: A = C(ic), i.e ic contains for each row in A the index of the unique row in C --> ic has the same length as A. 

% create marginal probability distribution of all unique predictor tuples   
    num_unique = length(ia);        % number of unique predictor tuples
    binlabels = [1:num_unique];     % as unique data tuples receive labels 1...num_tuples, 'binlabels' is a list of labels of all unique data tuples
    fs = hist(ic,binlabels);        % calculate frequencies of all unique data tuples
                                    % Note: All values in 'fs' are > 0 (no empty bins)
    ps = fs/sum(fs);                % normalize frequencies to probabilites
    ps = ps';                       % flip for further use

% for each unique predictor tuple, get all related target tuples, and calculate its entropy
    
    cHs = NaN(num_unique,1); % container for conditional entropies

    % loop over all unique predictor tuples
    for i = 1 : num_unique
        
        % find indices of all rows with predictor tuple 'i' and extract the corresponding rows from targets (conditional set)
        targets_cond = targets_binned(ic==i,:); 
        
        % calculate (conditional) entropy of the target(s) given the unique predictor tuple
        cHs(i) = f_entropy(targets_cond); 
        
    end

% calculate overall conditional entropy over all unique predictor tuples
    cH = sum(ps .* cHs); % probability-weighted sum of all conditional entropies (expected value)

end

