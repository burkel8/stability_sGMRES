function [x, res, berr, cndV, cndAV, sbnd, taulog, tlog] = sgmres_truncatedArnoldi_adpt(A, b, x0, M, tol, m, maxiters, t, tol_tau)
%% sGMRES with truncated Arnoldi, restart when tau gets too large
n = size(A, 1);
rng(1);            % seed random number generator
s = 2*(m+1);       % size of s
hS = srhtb2(n, s); % sketching operator

normA = norm(A, 'fro');
normb = norm(b);

r0 = M(b - A*x0);

V = zeros(n, m+1);
AV = zeros(n, m+1);
H = zeros(m+1,m);
SAV = [];
res = [];
berr = [];
cndV = [];
cndAV = [];
sbnd = [];
taulog = [];
tlog = [];
it = 0;
% dt = ceil(m*0.1);
t = 1;
taubar = 0;

while it < maxiters

    V(:, 1) = r0/norm(r0);
    SV = hS(V(:, 1));
    Sb = hS(r0);

    for j = 1:m
        tlog(end+1) = min(j, t);
        w = M(A*V(:,j));
        AV(:, j) = w;
        SAV(:,j) = hS(w);
        for i = max(1,j-t+1):j
            H(i,j) = V(:,i)'*w;
            w = w - H(i,j)*V(:,i);
        end
        H(j+1,j) = norm(w);
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = hS(V(:,j+1));

        [U, T] = qr(SAV(:, 1:j), 0);
        y = (T \ (U'*Sb));
        e = V(:,1:j) * y;
        Se = SV(:,1:j) * y;

        x = x0 + e;
        r = b - A*x;

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV = norm(SV);

        res(end+1) = normr;
        berr(end+1) = normr/(normA*normx + normb);
        cndV(end+1) = cond(SV(:, 1:j));
        cndAV(end+1) = cond(SAV(:, 1:j));
        sbnd(end+1) = (normV*normy)/normx;

        it = it + 1;

        if it >= maxiters
            break;
        end

        Sx = hS(x);
        tau = norm(SV(:, 1:j))*norm(y)/norm(e);
        taulog(end+1) = norm(V(:, 1:j))*norm(y)/norm(e);


        
        taubarold = taubar;
        taubar = norm(SV(:, 1:j))*normA*norm(y)/norm(SAV(:, 1:j)*y);
        if taubar*tol_tau >= 1
            if taubar > 1.1*taubarold
                t = min(j+1, 2*t);
            end
        end

        if berr(end) < tol
            return;
        end

    end

    x0 = x;
    r0 = M(r);

end

end