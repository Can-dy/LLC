% =========================================================================
%基于LLC特征的刑侦图像分类算法
% May, 2018.
% =========================================================================
clear all; close all; clc;

% -------------------------------------------------------------------------
% 第一步：设置参数
pyramid = [1, 2, 4];                % 金字塔空间结构快 
knn = 5;                            % 本地编码的邻近数
c = 10;                             % 在Liblinear包中，线性SVM的正则化参数

nRounds = 10;                       % 数据集随机测试次数
tr_num  = 30;                       % 每类训练例
mem_block = 3000;                   % 每次最多加载测试特征点

addpath('Liblinear/matlab');        % 我们使用Liblinear包，下载并编译matlab代码 

img_dir = 'image/刑侦图像7类-压缩后';       % 图像数据库目录
data_dir = 'data/刑侦图像7类-压缩后';       % 保存SIFT描述符的目录
fea_dir = 'features/刑侦图像7类-压缩后';    % 保存最终图像特征的目录

% -------------------------------------------------------------------------
%第二步：提取刑侦图像特征

%提取SIFT描述子，我们使用这个包中的Lazebnik教授的matlab代码，在函数“extr_sift”中更改SIFT提取的参数
% extr_sift(img_dir, data_dir);       % 特征提过一次，不需再提

database = retr_database_dir(data_dir);

if isempty(database),
    error('Data directory error!');
end
% -------------------------------------------------------------------------
%第三步：检索数据库的目录并加载码本
Bpath = ['dictionary/Caltech101_SIFT_Kmeans_1024.mat'];

load(Bpath);

nCodebook = size(B, 2);              %码本大小128*1024 

% -------------------------------------------------------------------------
% 第四步：基于字典与每张图像的一组SIFT特征，提取它的LLC特征
dFea = sum(nCodebook*pyramid.^2);   % 1024*21
nFea = length(database.path);       %700张

fdatabase = struct;
fdatabase.path = cell(nFea, 1);         % 每个图像特征的路径
fdatabase.label = zeros(nFea, 1);       %  每个图像特征的标签

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
for iter1 = 1:nFea,                  %打点显示运行过程
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
    
    %对每张图像的一组SIFT特征，提取它的LLC特征，保存在Feature目录中
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
% save  'F:\库\LLC2\ALLC1\LLC1\feadate\L1' L1;
% save  'F:\库\LLC2\ALLC1\LLC1\feadate\L2' L2;
% save  'F:\库\LLC2\ALLC1\LLC1\feadate\L3' L3;
% save  'F:\库\LLC2\ALLC1\LLC1\feadate\L4' L4;
% save  'F:\库\LLC2\ALLC1\LLC1\feadate\L5' L5;
% save  'F:\库\LLC2\ALLC1\LLC1\feadate\L6' L6;
% save  'F:\库\LLC2\ALLC1\LLC1\feadate\L7' L7;
 
% -------------------------------------------------------------------------
%第五步：基于Liblinear SVM与LLC特征，进行分类测试

% load('F:\库\LLC2\ALLC1\LLC1\feadate\L1.mat')
% load('F:\库\LLC2\ALLC1\LLC1\feadate\L2.mat')
% load('F:\库\LLC2\ALLC1\LLC1\feadate\L3.mat')
% load('F:\库\LLC2\ALLC1\LLC1\feadate\L4.mat')
% load('F:\库\LLC2\ALLC1\LLC1\feadate\L5.mat')
% load('F:\库\LLC2\ALLC1\LLC1\feadate\L6.mat')
% load('F:\库\LLC2\ALLC1\LLC1\feadate\L7.mat')


fprintf('\n Testing...\n');
clabel = unique(fdatabase.label);  %取集合fdatabase.label的不重复元素构成的向量
nclass = length(clabel);
accuracy = zeros(nRounds, 1);

for ii = 1:nRounds,                %随机运行多次，统计平均精度
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
    
    % 加载训练特征
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

    % 加载测试特征
    ts_num = length(ts_idx);
    ts_label = [];
    
    if ts_num < mem_block,
        % 将测试功能直接加载到内存中进行测试
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
        % 逐块加载测试特征
        num_block = floor(ts_num/mem_block);
        rem_fea = rem(ts_num, mem_block);
        
        curr_ts_fea = zeros(mem_block, dFea);
        curr_ts_label = zeros(mem_block, 1);
        
        C = [];
        
        for jj = 1:num_block,
            block_idx = (jj-1)*mem_block + (1:mem_block);
            curr_idx = ts_idx(block_idx); 
            
            % 加载当前的特征块
            for kk = 1:mem_block,
                fpath = fdatabase.path{curr_idx(kk)};
                load(fpath, 'fea', 'label');
                curr_ts_fea(kk, :) = fea';
                curr_ts_label(kk) = label;
            end    
            
            % 当前测试块
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
    
    % 通过对不同类进行平均来归一化分类精度
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

Ravg = mean(accuracy);                  % 平均识别率
Rstd = std(accuracy);                   % 识别率的标准方差

fprintf('===============================================');
fprintf('Average classification accuracy: %f\n', Ravg);
fprintf('Standard deviation: %f\n', Rstd);    
fprintf('===============================================');
    
