function [beta] = LLC_pooling(feaSet, B, pyramid, knn) 
% feaSet1*1 struct����feaArr��128*1680����x��1680*1����y��1680*1����width��300����height��225��   
% B �ֲ�Լ�����Ա�����뱾
% pyramid ϡ��������ṹ
% LLC������ڽ���
% beta ͼ��������� 
dSize = size(B, 2);
nSmp = size(feaSet.feaArr, 2);

img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

% llc ����
llc_codes = LLC_coding_appr(B', feaSet.feaArr', knn);
llc_codes = llc_codes';

% �ռ���
pLevels = length(pyramid);
% ÿ������Ŀո�
pBins = pyramid.^2;
% �ܿո�
tBins = sum(pBins);

beta = zeros(dSize, tBins);
bId = 0;

for iter1 = 1:pLevels,
    
    nBins = pBins(iter1);
    
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);
    
    % �ҵ�ÿ�����������������ĸ��ռ� 
    xBin = ceil(feaSet.x / wUnit);
    yBin = ceil(feaSet.y / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end      
        beta(:, bId) = max(llc_codes(:, sidxBin), [], 2);  %�ȽϾ�����У�������ֵ
    end
end

if bId ~= tBins,
    error('Index number error!');
end

beta = beta(:);
beta =beta./sqrt(sum(beta.^2));
