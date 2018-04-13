%function Gamma = CSC_L1_L2inf(imn, T, Omega, idxSt, idxPP, n, m, rDl, idxSl)
%%%function Gamma = CSC_L1_L2inf(imn, T, Omega, idxSt, idxPP, n, m)

[H, W] = size(imn);
y = gpuArray(imn(:));
%y = imn(:);

ppm = size(Omega,2);
pp = ppm/m;
jjSt = repmat(1:(H*W), [ppm, 1]);     % for stripe aggregation 
jjPP = repmat(1:(H*W), [prod(n), 1]); % for patch aggregation

% ADMM parameters
MAXITER = 500;
ABSTOL = 5e-3;
RELTOL = 5e-3;
rho = 1;

% Precomputations to accelerate QCQPs
cc = 200;
A0 = eye(ppm)*cc + Omega'*Omega;
[P, D0] = eig(A0);
D0 = diag(D0) - cc;
maxD0 = max(D0);
minD0 = min(D0);
Pt = P';
PtOmegat = Pt*Omega';
b = y(idxPP);
PtOmegatb = PtOmegat*b;
Pt = P';
OmegaP = Omega*P;

% Initialization
Gamma = gpuArray(zeros(m*H*W,1));
Gamma2 = gpuArray(zeros(ppm,H*W));
U = Gamma2;


r = 2; s = 2;
eps_r = 1; eps_s = 1;
for iter = 1:MAXITER
    tic
    
    % Checking stopping criterion
    fprintf('iter %d, r/eps_r = %.3f, s/eps_s = %.3f\n', iter-1, r/eps_r, s/eps_s);
    % fprintf('iter %d, r = %.3f, eps_r = %.3f, s = %.3f, eps_s = %.3f\n', iter-1, r, eps_r, s, eps_s);
    if (r < eps_r  && s < eps_s && iter > 1)
        fprintf('breaking\n');
        break;
    end
    xold = Gamma(idxSt);
    

    % UPDATE GAMMA (aggregation and soft-thresholding)
    a = Gamma2+U;
    WW = sparse(idxSt(:), jjSt(:), a(:)); 
    WW = sum(WW, 2);
    WW = full(WW);
    Gamma = bsxfun(@times, max(0, abs(WW) - 1/rho), sign(WW))/pp;
    
    % UPDATE Gamma2 (QCQPs)
    a = Gamma(idxSt) - U;
    Pta = Pt*a;
    lmin = 0*ones(1,H*W);
    lmax = 10*ones(1,H*W);
    it = 0;
    while it < 20
        it = it+1;
        lcurr = .5*(lmin + lmax);
        invlcurr = 1./lcurr;
        c = bsxfun(@times, Pta, invlcurr) + PtOmegatb;
        c = c./bsxfun(@plus, D0, invlcurr);
        xx = OmegaP*c - b;
        idx = find(sum(xx.^2) < T);
        %length(idx)/(H*W)
        tmp = lmin;
        lmin = lcurr;
        lmax(idx) = lcurr(idx);
        lmin(idx) = tmp(idx);
        % figure(333), clf, hold on, plot(sum(xx.^2)), plot(T*ones(1,H*W), '--'); pause(.1);
        % figure(222), clf, hold on, plot(lmin), plot(lmax), pause(.3);
    end
    invlcurr = 1./lmax;
    c = bsxfun(@times, Pta, invlcurr) + PtOmegatb;
    c = c./bsxfun(@plus, D0, invlcurr);
    Gamma2 = P*c;

    
    % Warning for l2inf criteria
    xx = Omega*Gamma2 - b;
    xx = sum(xx.^2);
    if (max(xx) > T)
        fprintf(1,'WARNING: l2inf constraint not enforced\n');
    end
    
    % Update dual variable
    U = U + Gamma2 - Gamma(idxSt);

    % Stopping criteria
    x = Gamma(idxSt);
    z = Gamma2;
    r = norm(x - z, 'fro');  % feasibility
    s = rho*norm(x-xold, 'fro'); % dual feasibility
    eps_r = ABSTOL + RELTOL * max( norm(x,'fro'), norm(z, 'fro') );
    eps_s = ABSTOL + RELTOL * rho*norm(U,'fro'); 
    
    % VISUALIZATION
    GammaT = reshape(Gamma, H, W, m);
    figure(16), spy(GammaT(:,:,4))
    
    % Image reconstruction (Optional)
    a = rDl*Gamma(idxSl);
    WW = sparse(idxPP(:), jjPP(:), a(:)); 
    WW = sum(WW, 2);
    WW = full(WW);
    figure(17), imagesc(reshape(WW, H, W)); colormap gray; axis equal;
    drawnow;
    
    %toc
end
