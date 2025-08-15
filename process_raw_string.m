% Function to create file_name, subject_prefix and setting
function [file_name, subjectPrefix, setting, save_file_name] = process_raw_string(folder, rawString)
    % Remove the '@' at the beginning
    rawString = erase(rawString, '@raw');

    % Extract the session number from the folder
    sessionMatch = regexp(folder, 'ses(\d+)', 'tokens');
    if isempty(sessionMatch)
        error('Session number not found in folder string.');
    end
    sessionNumber = sessionMatch{1}{1}; % Extract the first match (the number)
    
    % Construct the run number based on the session number
    runNumber = sprintf('run%s', sessionNumber); % e.g., 'run08'
    
    % Construct file_name with 'data_0raw_' prefix and the cleaned rawString
    file_name = ['data_0raw_', rawString]; % Prefix with 'data_0raw_'
    
    % Extract subject prefix based on 'PT' or 'PTN'
    if contains(rawString, 'PTN')
        subjectPrefix = regexp(rawString, 'PTN\d+', 'match'); % Match 'PTNXX'
    else
        subjectPrefix = regexp(rawString, 'PT\d+', 'match'); % Match 'PTXX'
    end
    
    if isempty(subjectPrefix)
        error('Subject prefix not found in rawString.');
    end
    subjectPrefix = subjectPrefix{1}; % Extract the first match

    % Determine the setting (TONIC or BURST) from folder
    if contains(folder, '_to')
        setting = 'TONIC';
    elseif contains(folder, '_bu')
        setting = 'BURST';
    else
        error('The folder string does not contain "_to" or "_bu".');
    end

    % Construct file_name with the desired format
    save_file_name = sprintf('FT data\\%s_%s_%s_FT_data', runNumber, subjectPrefix, setting);
end