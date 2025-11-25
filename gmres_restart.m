function [x, res, berr, cndV, cndAV, sbnd, taulog, tlog] = gmres_restart(A, b, x0, M, tol, m, nrestarts, t)
% x0 = zeros(n, 1);
normA = norm(A, 'fro');
normb = norm(b);

n = size(A, 1);
r0 = M(b - A*x0);
V = zeros(n, m+1);
AV = zeros(n, m+1);
H = zeros(m+1, m);
res = [];  % stores residuals
berr = []; % stores backward error
cndV = [];  % stores conditon number of V
cndAV = []; % stores condition number of A*V
sbnd = []; % stores sharper bound
taulog = [];
tlog = [];
it = 0;

for iterout = 1:nrestarts

    V(:, 1) = r0/norm(r0);

    for j = 1:m
        tlog(end+1) = j;
        w = M(A*V(:,j));
        AV(:, j) = w;
        for i = max(1,j-t+1):j
            % for reo = 0:0
            h = V(:,i)'*w;
            w = w - h*V(:,i);
            H(i,j) = h;
            % end
        end
        H(j+1,j) = norm(w);
        V(:,j+1) = w/H(j+1,j);

        y = H(1:j+1,1:j)\(norm(r0)*eye(j+1,1));
        e = V(:,1:j)*y;
        x = x0 + e;
        r = b - A*x;

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV = norm(V(:, 1:j));

        res(end+1) = normr;
        berr(end+1) = normr/(normA*normx + normb);
        cndV(end+1) = cond(V(:, 1:j));
        cndAV(end+1) = cond(AV(:, 1:j));
        sbnd(end+1) = (normV*normy)/normx;
        taulog(end+1) = normV*normy/norm(e);

        if berr(end) < tol
            return;
        end

    end

    x0 = x;
    r0 = M(r);
end
end