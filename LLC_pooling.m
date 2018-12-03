function [beta] = LLC_pooling(feaSet, B, pyramid, knn) 
% feaSet1*1 struct，有feaArr（128*1680）、x（1680*1）、y（1680*1）、width（300）、height（225）   
% B 局部约束线性编码的码本
% pyramid 稀疏金字塔结构
% LLC编码的邻近数
% beta 图像特征输出 
dSize = size(B, 2);
nSmp = size(feaSet.feaArr, 2);

img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

% llc 编码
llc_codes = LLC_coding_appr(B', feaSet.feaArr', knn);
llc_codes = llc_codes';

% 空间层次
pLevels = length(pyramid);
% 每个级别的空格
pBins = pyramid.^2;
% 总空格
tBins = sum(pBins);

beta = zeros(dSize, tBins);
bId = 0;

for iter1 = 1:pLevels,
    
    nBins = pBins(iter1);
    
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);
    
    % 找到每个本地描述符属于哪个空间 
    xBin = ceil(feaSet.x / wUnit);
    yBin = ceil(feaSet.y / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end      
        beta(:, bId) = max(llc_codes(:, sidxBin), [], 2);  %比较矩阵的行，输出最大值
    end
end

if bId ~= tBins,
    error('Index number error!');
end

beta = beta(:);
beta =beta./sqrt(sum(beta.^2));
