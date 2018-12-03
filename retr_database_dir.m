function [database] = retr_database_dir(rt_data_dir)
%=========================================================================
% inputs
% rt_data_dir   -the rootpath for the database. e.g. '../data/caltech101'
% outputs
% database      -a tructure of the dir
%                   .path   ÿ��ͼ���ļ���·��
%                   .label  ÿ��ͼ���ļ��ı�ǩ
%=========================================================================

fprintf('dir the database...');
subfolders = dir(rt_data_dir);    % '��'',���࣬9*1������name��date��bytes��isdir��datenum���fields

database = [];   %16*24

database.imnum = 0;                % ���ݿ����ͼ����
database.cname = {};               % ÿ������
database.label = [];               % ÿ���ǩ
database.path = {};                % ����ÿ�����ÿ��ͼ���·��
database.nclass = 0;               %16*24

for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
        database.nclass = database.nclass + 1;
        
        database.cname{database.nclass} = subname;
        
        frames = dir(fullfile(rt_data_dir, subname, '*.mat'));     % ÿ��100������Ϊ*.mat
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