% =========================================================================
%����LLC����������ͼ������㷨
% May, 2018.
% =========================================================================
clear all; close all; clc;

% -------------------------------------------------------------------------
% ��һ�������ò���
pyramid = [1, 2, 4];                % �������ռ�ṹ�� 
knn = 5;                            % ���ر�����ڽ���
c = 10;                             % ��Liblinear���У�����SVM�����򻯲���

nRounds = 10;                       % ���ݼ�������Դ���
tr_num  = 30;                       % ÿ��ѵ����
mem_block = 3000;                   % ÿ�������ز���������

addpath('Liblinear/matlab');        % ����ʹ��Liblinear�������ز�����matlab���� 

img_dir = 'image/����ͼ��7��-ѹ����';       % ͼ�����ݿ�Ŀ¼
data_dir = 'data/����ͼ��7��-ѹ����';       % ����SIFT��������Ŀ¼
fea_dir = 'features/����ͼ��7��-ѹ����';    % ��������ͼ��������Ŀ¼

% -------------------------------------------------------------------------
%�ڶ�������ȡ����ͼ������

%��ȡSIFT�����ӣ�����ʹ��������е�Lazebnik���ڵ�matlab���룬�ں�����extr_sift���и���SIFT��ȡ�Ĳ���
% extr_sift(img_dir, data_dir);       % �������һ�Σ���������

database = retr_database_dir(data_dir);

if isempty(database),
    error('Data directory error!');
end
% -------------------------------------------------------------------------
%���������������ݿ��Ŀ¼�������뱾
Bpath = ['dictionary/Caltech101_SIFT_Kmeans_1024.mat'];

load(Bpath);

nCodebook = size(B, 2);              %�뱾��С128*1024 

% -------------------------------------------------------------------------
% ���Ĳ��������ֵ���ÿ��ͼ���һ��SIFT��������ȡ����LLC����
dFea = sum(nCodebook*pyramid.^2);   % 1024*21
nFea = length(database.path);       %700��

fdatabase = struct;
fdatabase.path = cell(nFea, 1);         % ÿ��ͼ��������·��
fdatabase.label = zeros(nFea, 1);       %  ÿ��ͼ�������ı�ǩ

% L1=zeros(100,21505);
% L1num=1;
% L2=zeros(100,21505);
% L2num=1;
% L3=zeros(100,21505);
% L3num=1;
% L4=zeros(100,21505);
% L4num=1;
% L5=zeros(100,21505);
% L5num=1;
% L6=zeros(100,21505);
% L6num=1;
% L7=zeros(100,21505);
% L7num=1;
for iter1 = 1:nFea,                  %�����ʾ���й���
    if ~mod(iter1, 5),
       fprintf('.');
    end
    if ~mod(iter1, 100),
        fprintf(' %d images processed\n', iter1);
    end
    
    fpath = database.path{iter1};
    flabel = database.label(iter1);
    
    load(fpath);
    [rtpath, fname] = fileparts(fpath);
    feaPath = fullfile(fea_dir, num2str(flabel), [fname '.mat']);
    
    %��ÿ��ͼ���һ��SIFT��������ȡ����LLC������������FeatureĿ¼��
    fea = LLC_pooling(feaSet, B, pyramid, knn);
    label = database.label(iter1);
%     L1name=str2num(fname);
%     L2name=str2num(fname);
%     L3name=str2num(fname);
%     L4name=str2num(fname);
%     L5name=str2num(fname);
%     L6name=str2num(fname);
%     L7name=str2num(fname);
%     if label==1
%         L1(L1num,1)= L1name;
%         L1(L1num,2:21505)=fea;
%         L1num=L1num+1;
%     end
%      if label==2
%         L2(L2num,1)= L2name;
%         L2(L2num,2:21505)=fea;
%         L2num=L2num+1;
%      end
%      if label==3
%         L3(L3num,1)= L3name;
%         L3(L3num,2:21505)=fea;
%         L3num=L3num+1;
%      end
%      if label==4
%         L4(L4num,1)= L4name;
%         L4(L4num,2:21505)=fea;
%         L4num=L4num+1;
%      end
%       if label==5
%         L5(L5num,1)= L5name;
%         L5(L5num,2:21505)=fea;
%         L5num=L5num+1;
%       end
%       if label==6
%         L6(L6num,1)= L6name;
%         L6(L6num,2:21505)=fea;
%         L6num=L6num+1;
%       end
%      if label==7
%         L7(L7num,1)= L7name;
%         L7(L7num,2:21505)=fea;
%         L7num=L7num+1;
%      end
%      
    if ~isdir(fullfile(fea_dir, num2str(flabel))),
        mkdir(fullfile(fea_dir, num2str(flabel)));
    end      
    save(feaPath, 'fea', 'label');

    fdatabase.label(iter1) = flabel;
    fdatabase.path{iter1} = feaPath;
%     save(fdatabase,'-struct','flable','feaPath');
end
% save  'F:\��\LLC2\ALLC1\LLC1\feadate\L1' L1;
% save  'F:\��\LLC2\ALLC1\LLC1\feadate\L2' L2;
% save  'F:\��\LLC2\ALLC1\LLC1\feadate\L3' L3;
% save  'F:\��\LLC2\ALLC1\LLC1\feadate\L4' L4;
% save  'F:\��\LLC2\ALLC1\LLC1\feadate\L5' L5;
% save  'F:\��\LLC2\ALLC1\LLC1\feadate\L6' L6;
% save  'F:\��\LLC2\ALLC1\LLC1\feadate\L7' L7;
 
% -------------------------------------------------------------------------
%���岽������Liblinear SVM��LLC���������з������

% load('F:\��\LLC2\ALLC1\LLC1\feadate\L1.mat')
% load('F:\��\LLC2\ALLC1\LLC1\feadate\L2.mat')
% load('F:\��\LLC2\ALLC1\LLC1\feadate\L3.mat')
% load('F:\��\LLC2\ALLC1\LLC1\feadate\L4.mat')
% load('F:\��\LLC2\ALLC1\LLC1\feadate\L5.mat')
% load('F:\��\LLC2\ALLC1\LLC1\feadate\L6.mat')
% load('F:\��\LLC2\ALLC1\LLC1\feadate\L7.mat')


fprintf('\n Testing...\n');
clabel = unique(fdatabase.label);  %ȡ����fdatabase.label�Ĳ��ظ�Ԫ�ع��ɵ�����
nclass = length(clabel);
accuracy = zeros(nRounds, 1);

for ii = 1:nRounds,                %������ж�Σ�ͳ��ƽ������
    fprintf('Round: %d...\n', ii);
    tr_idx = [];
    ts_idx = [];
    
    for jj = 1:nclass,
        idx_label = find(fdatabase.label == clabel(jj));
        num = length(idx_label);
        
        idx_rand = randperm(num);
        
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
    end
    
    fprintf('Training number: %d\n', length(tr_idx));
    fprintf('Testing number:%d\n', length(ts_idx));
    
    % ����ѵ������
    tr_fea = zeros(length(tr_idx), dFea);
    tr_label = zeros(length(tr_idx), 1);
    
    for jj = 1:length(tr_idx),
        fpath = fdatabase.path{tr_idx(jj)};
        load(fpath, 'fea', 'label');
        tr_fea(jj, :) = fea';
        tr_label(jj) = label;
    end
    
    options = ['-c ' num2str(c)];
    model = train(double(tr_label), sparse(tr_fea), options);
    clear tr_fea;

    % ���ز�������
    ts_num = length(ts_idx);
    ts_label = [];
    
    if ts_num < mem_block,
        % �����Թ���ֱ�Ӽ��ص��ڴ��н��в���
        ts_fea = zeros(length(ts_idx), dFea);
        ts_label = zeros(length(ts_idx), 1);

        for jj = 1:length(ts_idx),
            fpath = fdatabase.path{ts_idx(jj)};
            load(fpath, 'fea', 'label');
            ts_fea(jj, :) = fea';
            ts_label(jj) = label;
        end

        [C] = predict(ts_label, sparse(ts_fea), model);
    else
        % �����ز�������
        num_block = floor(ts_num/mem_block);
        rem_fea = rem(ts_num, mem_block);
        
        curr_ts_fea = zeros(mem_block, dFea);
        curr_ts_label = zeros(mem_block, 1);
        
        C = [];
        
        for jj = 1:num_block,
            block_idx = (jj-1)*mem_block + (1:mem_block);
            curr_idx = ts_idx(block_idx); 
            
            % ���ص�ǰ��������
            for kk = 1:mem_block,
                fpath = fdatabase.path{curr_idx(kk)};
                load(fpath, 'fea', 'label');
                curr_ts_fea(kk, :) = fea';
                curr_ts_label(kk) = label;
            end    
            
            % ��ǰ���Կ�
            ts_label = [ts_label; curr_ts_label];
            [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model);
            C = [C; curr_C];
        end
        
        curr_ts_fea = zeros(rem_fea, dFea);
        curr_ts_label = zeros(rem_fea, 1);
        curr_idx = ts_idx(num_block*mem_block + (1:rem_fea));
        
        for kk = 1:rem_fea,
           fpath = fdatabase.path{curr_idx(kk)};
           load(fpath, 'fea', 'label');
           curr_ts_fea(kk, :) = fea';
           curr_ts_label(kk) = label;
        end  
        
        ts_label = [ts_label; curr_ts_label];
        [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model); 
        C = [C; curr_C];        
    end
    
    % ͨ���Բ�ͬ�����ƽ������һ�����ྫ��
    acc = zeros(nclass, 1);

    for jj = 1 : nclass,
        c = clabel(jj);
        idx = find(ts_label == c);
        curr_pred_label = C(idx);
        curr_gnd_label = ts_label(idx);    
        acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
    end

    accuracy(ii) = mean(acc); 
    fprintf('Classification accuracy for round %d: %f\n', ii, accuracy(ii));
end

Ravg = mean(accuracy);                  % ƽ��ʶ����
Rstd = std(accuracy);                   % ʶ���ʵı�׼����

fprintf('===============================================');
fprintf('Average classification accuracy: %f\n', Ravg);
fprintf('Standard deviation: %f\n', Rstd);    
fprintf('===============================================');
    
