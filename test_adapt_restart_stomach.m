% test_adapt_restart_stomach
% Tests various sGMRES adaptive restart strategies using the stomach matrix.

% To reproduce Fig 4 in the paper, first run test_adapt_restart_stomach.m,
% then test_adapt_restart_fs_760_1.m, then test_adaptive_restart_shermann.m

clear all
close all
clc

% load matrix
load('stomach.mat');
A = Problem.A;

nzeros = nonzeros(A);
normA = norm(nzeros);

n = size(A,1);

A = @(x) A*x;

% seed random number generator to reproduce results
rng(1);

% construct right hand side
b = randn(n,1);
b = b/norm(b);
normb = norm(b);

nrestarts = 4 ; % max number of restarts
m = 150;        % number of Arnoldi iterations per cycle
maxiters = nrestarts*m; % max number of iterations
t = 2;         % Arnoldi truncation parameter
tol = 1e-15;   % convergence tolerance

rng(1);            % seed random number generator
s = 2*(m+1);       % size of s
hS = srhtb2(n, s); % sketching operator

%% sGMRES with truncated Arnoldi, restart when tau gets too large
x0 = zeros(n,1);
r0 = b;

AV2 = [];
H2 = zeros(m+1,m);
SAV2 = [];
res2 = []; berr2 = [];	cnd2 = []; cnd22 = []; sbnd2 = [];
it = 0;

while it < maxiters

    V2 = r0/norm(r0);
    SV2 = hS(V2);
    Sb = hS(r0);

    for j = 1:m
        w = A(V2(:,j));
        AV2(:, j) = w;
        SAV2(:,j) = hS(w);
        for i = max(1,j-t+1):j
            H2(i,j) = V2(:,i)'*w;
            w = w - H2(i,j)*V2(:,i);
        end
        H2(j+1,j) = norm(w);
        V2(:,j+1) = w/H2(j+1,j);
        SV2(:,j+1) = hS(V2(:,j+1));

        [U, T] = qr(SAV2(:, 1:j), 0);
        y = (T \ (U'*Sb));
        e = V2(:,1:j) * y;

        x = x0 + e;
        r = b - A(x);

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV2 = norm(SV2);

        res2(end+1) = norm(r);
        berr2(end+1) = norm(r)/(normA*normx + normb);
        cnd2(end+1) = cond(SV2);
        cnd22(end+1) = cond(SAV2);
        sbnd2(end+1) = (normV2*normy)/normx;

        it = it + 1;

        if it >= maxiters
            break;
        end

        Sx = hS(x);
        tau = norm(SV2)*norm(y)/norm(Sx);

        if(tau*eps > 1)
            break;
        end

        if berr2(end) < tol
            break;
        end

    end

    x0 = x;
    r0 = r;

end

%% sGMRES with truncated Arnoldi, restart when basis condition number gets too large
x0 = zeros(n,1);
r0 = b;

AV3 = [];
H3 = zeros(m+1,m);
SAV3 = [];
res3 = []; berr3 = [];	cnd3 = []; cnd33 = []; sbnd3 = [];
it = 0;

while it < maxiters

    V3 = r0/norm(r0);
    SV3 = hS(V3);
    Sb = hS(r0);

    for j = 1:m
        w = A(V3(:,j));
        AV3(:, j) = w;
        SAV3(:,j) = hS(w);
        for i = max(1,j-t+1):j
            H3(i,j) = V3(:,i)'*w;
            w = w - H3(i,j)*V3(:,i);
        end
        H3(j+1,j) = norm(w);
        V3(:,j+1) = w/H3(j+1,j);
        SV3(:,j+1) = hS(V3(:,j+1));

        [U, T] = qr(SAV3(:, 1:j), 0);
        y = (T \ (U'*Sb));
        e = V3(:,1:j) * y;

        x = x0 + e;
        r = b - A(x);

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV3 = norm(SV3);

        res3(end+1) = norm(r);
        berr3(end+1) = norm(r)/(normA*normx + normb);
        cnd3(end+1) = cond(SV3);
        cnd33(end+1) = cond(SAV3);
        sbnd3(end+1) = (normV3*normy)/normx;

        it = it + 1;
        
        if it >= maxiters
            break;
        end

        if(cond(SV3)*eps > 1)
            break;
        end

        if berr3(end) < tol
            break;
        end

    end

    x0 = x;
    r0 = r;

end

%% sGMRES with truncated Arnoldi, restart when tau gets too large, increase t
x0 = zeros(n,1);
r0 = b;

AV1 = [];
H1 = zeros(m+1,m);
SAV1 = [];
res1 = []; berr1 = [];	cnd1 = []; cnd11 = []; sbnd1 = [];
it = 0;

while it < maxiters

    V1 = r0/norm(r0);
    SV1 = hS(V1);
    Sb = hS(r0);

    for j = 1:m
        w = A(V1(:,j));
        AV1(:, j) = w;
        SAV1(:,j) = hS(w);
        for i = max(1,j-t+1):j
            H1(i,j) = V1(:,i)'*w;
            w = w - H1(i,j)*V1(:,i);
        end
        H1(j+1,j) = norm(w);
        V1(:,j+1) = w/H1(j+1,j);
        SV1(:,j+1) = hS(V1(:,j+1));

        [U, T] = qr(SAV1(:, 1:j), 0);
        y = (T \ (U'*Sb));
        e = V1(:,1:j) * y;

        x = x0 + e;
        r = b - A(x);

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV1 = norm(SV1);

        res1(end+1) = norm(r);
        berr1(end+1) = norm(r)/(normA*normx + normb);
        cnd1(end+1) = cond(SV1);
        cnd11(end+1) = cond(SAV1);
        sbnd1(end+1) = (normV1*normy)/normx;

        it = it + 1;

        if it >= maxiters
            break;
        end

        Sx = hS(x);
        tau = norm(SV1)*norm(y)/norm(Sx);

        if(tau*eps > 1)
            t = 2*t;
            break;
        end

        if berr1(end) < tol
            break;
        end

    end

    x0 = x;
    r0 = r;

end

%% sGMRES with truncated Arnoldi, restart when basis gets too large, and adaptively restart
x0 = zeros(n,1);
r0 = b;

AV4 = [];
H4 = zeros(m+1,m);
SAV4 = [];
res4 = []; berr4 = [];	cnd4 = []; cnd44 = []; sbnd4 = [];
it = 0;

while it < maxiters

    V4 = r0/norm(r0);
    SV4 = hS(V4);
    Sb = hS(r0);

    for j = 1:m
        w = A(V4(:,j));
        AV4(:, j) = w;
        SAV4(:,j) = hS(w);
        for i = max(1,j-t+1):j
            H4(i,j) = V4(:,i)'*w;
            w = w - H4(i,j)*V4(:,i);
        end
        H4(j+1,j) = norm(w);
        V4(:,j+1) = w/H4(j+1,j);
        SV4(:,j+1) = hS(V4(:,j+1));

        [U, T] = qr(SAV4(:, 1:j), 0);
        y = (T \ (U'*Sb));
        e = V4(:,1:j) * y;

        x = x0 + e;
        r = b - A(x);

        it = it + 1;
        
        if it >= maxiters
            break;
        end

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV4 = norm(SV4);

        res4(end+1) = norm(r);
        berr4(end+1) = norm(r)/(normA*normx + normb);
        cnd4(end+1) = cond(SV4);
        cnd44(end+1) = cond(SAV4);
        sbnd4(end+1) = (normV4*normy)/normx;

        if(cond(SV4)*eps > 1)
            t = 2*t;
            break;
        end

        if berr4(end) < tol
            break;
        end

    end

    x0 = x;
    r0 = r;

end


%% sGMRES with truncated Arnoldi, restart when tau gets too large, and adaptively restart
%% if basis condition number gets too large
x0 = zeros(n,1);
r0 = b;

AV6 = [];
H6 = zeros(m+1,m);
SAV6 = [];
res6 = []; berr6 = [];	cnd6 = []; cnd66 = []; sbnd6 = [];
it = 0;

while it < maxiters

    V6 = r0/norm(r0);
    SV6 = hS(V6);
    Sb = hS(r0);

    for j = 1:m
        w = A(V6(:,j));
        AV6(:, j) = w;
        SAV6(:,j) = hS(w);
        for i = max(1,j-t+1):j
            H6(i,j) = V6(:,i)'*w;
            w = w - H6(i,j)*V6(:,i);
        end
        H6(j+1,j) = norm(w);
        V6(:,j+1) = w/H6(j+1,j);
        SV6(:,j+1) = hS(V6(:,j+1));

        [U, T] = qr(SAV6(:, 1:j), 0);
        y = (T \ (U'*Sb));
        e = V6(:,1:j) * y;

        x = x0 + e;
        r = b - A(x);

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV6 = norm(SV6);

        res6(end+1) = norm(r);
        berr6(end+1) = norm(r)/(normA*normx + normb);
        cnd6(end+1) = cond(SV6);
        cnd66(end+1) = cond(SAV6);
        sbnd6(end+1) = (normV6*normy)/normx;

        it = it + 1;
        
        if it >= maxiters
            break;
        end

        Sx = hS(x);
        tau = norm(SV6)*norm(y)/norm(Sx);

        if(tau*eps > 1)
            if(cond(SV6)*eps > 1)
                t = 2*t;
            end
            break;
        end

        if j == m
            t=2*t;
        end

        if berr6(end) < tol
            break;
        end

    end

    x0 = x;
    r0 = r;

end

% standard restarted GMRES
x0 = zeros(n,1);
r0 = b;
AV5 = [];
H5 = zeros(m+1,m);
res5 = [];  % stores residuals
berr5 = []; % stores backward error
cnd5 = [];  % stores conditon number of V
cnd55 = []; % stores condition number of A*V
sbnd5 = []; % stores sharper bound
it = 0;

for i = 1:nrestarts

    V5 = r0/norm(r0);

    for j = 1:m
        w = A(V5(:,j));
        AV5(:, j) = w;
        for i = 1:j
            for reo = 0:0
                h = V5(:,i)'*w;
                w = w - h*V5(:,i);
                H5(i,j) = h;
            end
        end
        H5(j+1,j) = norm(w);
        V5(:,j+1) = w/H5(j+1,j);

        y = H5(1:j+1,1:j)\(norm(r0)*eye(j+1,1));
        e = V5(:,1:j)*y;
        x = x0 + e;
        r = b - A(x);

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV5 = norm(V5);

        res5(end+1) = normr;
        berr5(end+1) = normr/(normA*normx + normb);
        cnd5(end+1) = cond(V5);
        cnd55(end+1) = cond(AV5);
        sbnd5(end+1) = (normV5*normy)/normx;

        if berr5(end) < tol
            break;
        end

    end

    x0 = x;
    r0 = r;
end

figure

t = tiledlayout(1,3);

nexttile
semilogy(berr2,'LineWidth',3); hold on;
semilogy(berr3, 'LineWidth',3);
semilogy(berr1, 'LineWidth',3, 'Marker', 'o');
semilogy(berr4, 'LineWidth',3);
semilogy(berr6,'LineWidth',3 );
semilogy(berr5,'--','LineWidth',3 );

xlabel("iteration",'interpreter','latex');
ylabel("backward error", 'interpreter','latex');
ylim([1e-15 1e0])
yticks([1e-15 1e-10 1e-5 1e0]);
set(gca,'FontSize', 15, 'FontWeight', 'normal')

set(gcf,'Units','pixels','Position',[100 100 1000 300]);

