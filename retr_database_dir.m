function [database] = retr_database_dir(rt_data_dir)
%=========================================================================
% inputs
% rt_data_dir   -the rootpath for the database. e.g. '../data/caltech101'
% outputs
% database      -a tructure of the dir
%                   .path   每个图像文件的路径
%                   .label  每个图像文件的标签
%=========================================================================

fprintf('dir the database...');
subfolders = dir(rt_data_dir);    % '，'',七类，9*1，包含name，date，bytes，isdir，datenum五个fields

database = [];   %16*24

database.imnum = 0;                % 数据库的总图像数
database.cname = {};               % 每类名称
database.label = [];               % 每类标签
database.path = {};                % 包含每个类的每个图像的路径
database.nclass = 0;               %16*24

for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
        database.nclass = database.nclass + 1;
        
        database.cname{database.nclass} = subname;
        
        frames = dir(fullfile(rt_data_dir, subname, '*.mat'));     % 每类100个。命为*.mat
        c_num = length(frames);
                    
        database.imnum = database.imnum + c_num;
        database.label = [database.label; ones(c_num, 1)*database.nclass];
        
        for jj = 1:c_num,
            c_path = fullfile(rt_data_dir, subname, frames(jj).name);
            database.path = [database.path, c_path];
        end;    
    end;
end;
disp('done!');