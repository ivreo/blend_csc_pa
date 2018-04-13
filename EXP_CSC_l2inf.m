clear all;
close all;

if 1
    addpath mexfiles;
    addpath ompbox10;
    addpath image_helpers;
    addpath(genpath('./vlfeat'));
    addpath(genpath('ksvd'));
    addpath(genpath('spams-matlab'));
    vl_setup();
    addpath('matlab2tikz/src/')
    addpath(genpath('~/proj/main (copy)/matlab/'))
    addpath(genpath('~/proj/main (copy)/matlab/mex/')) 
    %set(0,'DefaultAxesFontSize',40)
end
    
% copying and checking exp_CSC_l2inf_B.m


% Loading signal
im = imresize(im2double(imread('lena.png')),.2);
%im = im(20:55,20:55);
%im = zeros(10);
im = im.';
[H,W] = size(im),

% ODCT dictionary (2D)
n = [5,5];
m =  25;
rDl = odct2dict(n, m);
rDl(:,1) = []; % no DC
rDl = rDl +.1;
rDl = rDl*diag(1./sqrt(sum(rDl.*rDl))); % normalization
m = size(rDl, 2);
Dl = reshape(rDl, n(1), n(2), m);
Dl = permute(Dl, [2,1,3]);

% Omega construction (2D)
Omega = [];
for  j = (-(n(2)-1)):(n(2)-1)
    jd = max(1-j, 1):min(n(2)-j, n(2));
    for i = (-(n(1)-1)):(n(1)-1)
        id = max(1-i,1) : min(n(1)-i,n(1));
        D = zeros(size(Dl));
        D(n(1)+1-id, n(2)+1-jd, :) = Dl(id, jd, :);
        %figure(55), imagesc(D(:,:,1)), pause(.01); % DEBUG
        Omega = cat(2, Omega, reshape(D, prod(n), m));
        %figure(56), imagesc(Omega), pause(.01); % DEBUG
    end
end

%break
% Slice extraction from Gamma
% equivalent of for j<W / for i<H / for k<m
idxSl = reshape( 1:(m*H*W), m, H*W);

% Stripe (2D)
p1 = 2*n(1)-1;
p2 = 2*n(2)-1;
idxSt = zeros(p1*p2*m, H*W);
for j = 1:W
    for i = 1:H
        l = (j-1)*H + i; % index of a patch
        % DD = zeros(H,W); % DEBUG
        % DD(i,j) = 1;     % DEBUG
        for jp = (-(n(2)-1)):(n(2)-1)
            js = rem(j+jp + W -1, W) +1;
            for ip = (-(n(1)-1)):(n(1)-1)
                is = rem(i+ip + H -1, H) +1;
                ls = (js-1)*H +is;
                ks = ((n(2)-1+jp)*p1 + (n(1)-1+ip))*m + (1:m);
                idxSt(ks, l) = idxSl(:,ls);
                % DD(is, js) = DD(is, js) + 1; % DEBUG
            end
        end
        % figure(57), imagesc(DD), pause(.01); % DEBUG
    end
end



% Patch extraction
idxPP = zeros(prod(n), H*W);
[js, is] = meshgrid(1:n(2),1:n(1) );
js = js(:); is = is(:);
for j = 1:W
    for i = 1:H
       i_idx = rem( i+ is -2, H) +1;
       j_idx = rem( j+ js -2, W) +1;
       idxPP(:,(j-1)*H+i) = (j_idx-1)*H+i_idx;
    end
end


% bar n^2 Dl;
I = eye(p1*p2*m);
J = prod(n)*I;
J(:, 1:((p1*(n(2)-1) + n(1)-1)*m)) = 0;
J(:, ((p1*(n(2)-1) + n(1))*m+1):end) = 0;
barDl = Omega*J;



% Noise
stdnoise = 20/255;
imn = im + stdnoise*rand(size(im));
T = prod(n)*stdnoise^2;
%T = 10*T;
%T = T/10

% Removing DC (2D)
filter = ones(n);
imndc = imfilter(imn, filter, 'replicate');
imnhf = imn - imndc;
%imn = imnhf;

%Gamma = CSC_l1_l2inf(imn, T, Omega, idxSt, idxPP, n, m);

Omega = barDl;

Gamma = CSC_L1_L2inf(imn, T, Omega, idxSt, idxPP, n, m, rDl, idxSl);



