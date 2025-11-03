function [data_binned] = f_binme(data,edges,include_outliers)
% Returns values in 'data' classified/binned/discretized by the bins provided in 'edges'
% Note: This task can be done by using 'histcounts' or 'discretize'. According to Matlab help, use histcounts to find the number of elements in each bin, 
%       and discretize to find which bin each element belongs to (which is the case here). Computation time differences are small, though. 
% Input
% - data: [num_data, num_dim, num_ens] matrix, where num_data is the number of data tuples (= sample size), 
%   num_dim is the number of dimensions of the data set (= number of variables), 
%   and num_ens is the number of ensemble members (= number of values for each variable in a particular data tuple)
%   - data must be NaN-free
%   - the third dimension is optional (no problem if absent)
% - edges [1,num_dim] cell array, with arrays of bin edges for each dimension
%   - the bins of each dimension must completely cover the corresponding values in 'data'
% - include_outliers: [1,1] boolean. Optional. Default ist 'false'. If 'false', then values outside the binning range will be binned to 'NaN', 
%                                                                   if 'true', they will be assigned to the smallest or largest bin
% Output
% - data_binned [num_data, num_dim, num_ens] matrix (same as 'data'), with strictly positive integers, indicating the bin number into which the original data were classified
% Dependencies
% - none
% Version
% - 2024/02/01 Uwe Ehret: Added optional argument 'include_outsiders'
% - 2022/03/29 Uwe Ehret: Added handling the 'ensemble' dimension in 'data'
% - 2022/03/09 Uwe Ehret: Initial version

% check if optional argument for outlier handling was passed
  if (~exist('include_outliers','var'))
    include_outliers = false;
  end

% get dimensions and initialize output
  [num_data, num_dim, num_ens] = size(data);  % number of time steps, number of variables, number of ensemble members        
  data_binned = NaN(num_data, num_dim, num_ens); % initialize output

% reshape the data (remove the ensemble dimension)
% Note: This was used in an older version, but not needed any more
%     dummy = permute(data,[1 3 2]);        % swap the 2nd dimension (variables) and 3d dimension (ensemble members)
%     data_reshaped = reshape(dummy,[],num_dim);   % remove the ensemble dimension, glue them to the lower end of the 2-d matrix (time steps, variables)
    
% check input data for NaN
  if any(isnan(data),'all')   
      error('input data contain NaN');
  end

% check if input data fall outside the bin edges
  if include_outliers % include outliers --> if there are any, put them in the smallest or largest bin 
    
    for i = 1 : num_dim     % loop over all variables
      dummy = data(:,i,:);
      dummy(dummy < edges{i}(1)) = edges{i}(1);     % replace all values below the leftmost edge with the value of that edge 
      dummy(dummy > edges{i}(end)) = edges{i}(end); % replace all values above the rightmost edge with the value of that edge 
      data(:,i,:) = dummy;
    end
  
  else % do not include outliers --> if there are any, return an error 
  
    for i = 1 : num_dim     % loop over all variables
        if min(data(:,i,:),[],'all') < edges{i}(1) 
            error(strcat("there are values in dim ", num2str(i), " < lowermost bin edge"));
        end
        if max(data(:,i,:),[],'all') > edges{i}(end) 
            error(strcat("there are values in dim ", num2str(i), " > uppermost bin edge"));
        end
    end
  
  end

% discretize the values (replace values by the number of the bin they fall into)
  for i = 1 : num_dim     % loop over all variables
      data_binned(:,i,:) = discretize(data(:,i,:),edges{i});
  end

end