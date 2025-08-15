%% Export Brainstorm files to Fieldtrip
% Caroline Witstok
% 30-10-2024

addpath(genpath('C:\Users\carow\Downloads\KT\Programma''s\brainstorm_241014\brainstorm3\toolbox'));

% Input strings
folder = 'PT06c_ses01_to';
rawString = '@rawsub-PT06c_ses-20190329_task-rest_run-01_meg_notch';

% Process rawString using function
[file_name, subjectPrefix, setting, save_file_name] = process_raw_string(folder, rawString);

save_file_name = 'FT data\PT06c_TONIC_run01_FT_data'

% Construct subject_name
subject_name = [subjectPrefix, '_', setting];

% Display results
disp(['subject_name = ', subject_name]);
disp(['file_name = ', file_name]);
disp(['save_file_name = ', save_file_name]);

% Specifiy path to raw data and channel files
base_path = 'F:\MEG GNN\brainstorm_db\';
protocol_path = 'PreprocessingGNN\data\';
subject_path = sprintf('%s\\', folder);

% Specify the subdirectory for raw data
raw_subdir = sprintf('%s\\', rawString)

% Specify the raw data file name
raw_file_name = sprintf('%s.mat', file_name)

% Specify the channel file name
channel_file_name = 'channel_ctf_acc1.mat';

% Construct the full paths
raw_path = fullfile(base_path, protocol_path, subject_path, raw_subdir, raw_file_name)
channel_path = fullfile(base_path, protocol_path, subject_path, raw_subdir, channel_file_name)

% Specify the sensor types
SensorTypes = 'MEG';

% Export to FieldTrip format
[ftData, DataMat, ChannelMat, iChannels] = out_fieldtrip_data(raw_path, channel_path, SensorTypes, 1);

% Save FieldTrip data structure
save(save_file_name, 'ftData', '-v7.3');