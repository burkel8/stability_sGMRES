function [x, res, berr, cndV, cndAV, sbnd, taulog, tlog] = sgmres_ssa_restart(A, b, x0, tol, m, maxiters, t)
n = size(A, 1);
rng(1);            % seed random number generator
s = 2*(m+1);       % size of s

if s <= n
    hS = srhtb2(n, s); % sketching operator
else
    hS = @(x) x;
end

normA = norm(A, 'fro');
normb = norm(b);

r0 = b - A*x0;

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

while it < maxiters

    V(:, 1) = r0/norm(r0);
    SV = hS(V(:, 1));
    Sb = hS(r0);

    for j = 1:m
        w = A*V(:,j);
        AV(:, j) = w;
        sw = hS(w); 
        SAV(:,j) = sw;
        [Q, R] = qr(SV(:, 1:j), 0);
        coeffs = R \ (Q'*sw);
        [~,ind] = maxk(abs(coeffs),t);
        w = w - V(:,ind)*coeffs(ind);
        H(ind,j) = coeffs(ind);
        sw = hS(w);
        H(j+1,j) = norm(sw);
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = sw/H(j+1,j);

        [U, T] = qr(SAV(:, 1:j), 0);
        y = (T \ (U'*Sb));
        e = V(:,1:j) * y ;
        x = x0 + e;
        r = b - A*x;

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV = norm(SV);

        res(end+1) = normr;
        berr(end+1) = normr/(normA*normx + normb);
        cndV(end+1) = cond(SV);
        cndAV(end+1) = cond(SAV);
        sbnd(end+1) = (normV*normy)/normx;

        it = it + 1;

        if it >= maxiters
            return;
        end

        tau = norm(SV(:, 1:j))*norm(y)/norm(e);

        taulog(end+1) = tau;

        if berr(end) < tol
            return;
        end

    end

    x0 = x;
    r0 = r;

end

end