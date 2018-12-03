% ========================================================================
% USAGE: [Coeff]=LLC_coding_appr(B,X,knn,lambda)
% Approximated Locality-constraint Linear Coding
%
% Inputs
%       B       -M x d codebook, M entries in a d-dim space
%       X       -N x d matrix, N data points in a d-dim space
%       knn     -number of nearest neighboring
%       lambda  -regulerization to improve condition
%
% Outputs
%       Coeff   -N x M matrix, each row is a code for corresponding X
% ========================================================================

function [Coeff] = LLC_coding_appr(B, X, knn, beta)
% ���ƾֲ�Լ�����Ա���
% 
if ~exist('knn', 'var') || isempty(knn),
    knn = 5;
end

if ~exist('beta', 'var') || isempty(beta),
    beta = 1e-4;
end

nframe=size(X,1);
nbase=size(B,1);

%Ѱ���������
XX = sum(X.*X, 2);
%��ӦԪ����ˣ���������
BB = sum(B.*B, 2);
D  = repmat(XX, 1, nbase)-2*X*B'+repmat(BB', nframe, 1);
%ŷ�Ͼ���
%B=repmat([1,2;3,4],2,3)
%B=1 2 1 2 1 2
%  3 4 3 4 3 4          4*6ά
%  1 2 1 2 1 2
%  3 4 3 4 3 4
IDX = zeros(nframe, knn);
for i = 1:nframe,
	d = D(i,:);
	[dummy, idx] = sort(d, 'ascend');  
%�˴�sort������d������������
	IDX(i, :) = idx(1:knn);
end

% llc approximation coding
II = eye(knn, knn);    
%�˴�eye��������5*5��λ����
Coeff = zeros(nframe, nbase);
for i=1:nframe
   idx = IDX(i,:);
   z = B(idx,:) - repmat(X(i,:), knn, 1);           % shift ith pt to origin
   C = z*z';                                        % Э������
   C = C + II*beta*trace(C);                        % ��~ci regularlization (K>D)
%trace�����Խ�Ԫ��֮��
   w = C\ones(knn,1);
   w = w/sum(w);                                    % enforce sum(w)=1
   Coeff(i,idx) = w';
end
