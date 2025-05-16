% test1_stomach
% Plots the backward error curves for sGMRES applied to the stomach matrix.

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

nrestarts = 5; % max number of restarts
m = 150;       % number of Arnoldi iterations
t = 3;         % Arnoldi truncation parameter

% The sketching size for non-restarted implementation
s = 2*(m*nrestarts+1);
rng(1);
hS = srhtb2(n, s); % sketching operator

%% sGMRES with truncated Arnoldi (no restarts)
x0 = zeros(n,1);
r0 = b;

AV1 = [];   % stores A*V, for basis V
H1 = zeros(m+1,m);  % Hessenberg matrix
SAV1 = [];          % stores S*A*V
res1 = [];          % stores residual norms
berr1 = [];	        % stores backward error
cnd1 = [];          % stores condition number of V
cnd11 = [];         % stores condition number of A*V
sbnd1 = [];         % stores sharper upper bound

V1 = r0/norm(r0);
SV1 = hS(V1);
Sb = hS(r0);

for j = 1:m*nrestarts
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
    e = V1(:,1:j) * y ;

    x = x0 + e;
    r = b - A(x);

    normx = norm(x);
    normr = norm(r);
    normy = norm(y);
    normV1 = norm(SV1);

    res1(end+1) = normr;
    berr1(end+1) = normr/(normA*normx + normb);
    cnd1(end+1) = cond(SV1);
    cnd11(end+1) = cond(SAV1);
    sbnd1(end+1) = (normV1*normy)/normx;
end


%% sGMRES with sketch-and-select Arnoldi (no restarts)
x0 = zeros(n,1);
r0 = b;

AV0 = [];
H0 = zeros(m+1,m);
SAV0 = [];
res0 = []; berr0 = []; cnd0 = []; cnd00 = []; sbnd0 = [];

V0 = r0/norm(r0);
SV0 = hS(V0);
Sb = hS(r0);

for j = 1:m*nrestarts
    w = A(V0(:,j));
    AV0(:, j) = w;
    sw = hS(w); SAV0(:,j) = sw;
    [Q, R] = qr(SV0(:, 1:j), 0);
    coeffs = R \ (Q'*sw);
    [~,ind] = maxk(abs(coeffs),t);
    w = w - V0(:,ind)*coeffs(ind);
    H0(ind,j) = coeffs(ind);
    sw = hS(w);
    H0(j+1,j) = norm(sw);
    V0(:,j+1) = w/H0(j+1,j);
    SV0(:,j+1) = sw/H0(j+1,j);

    [U, T] = qr(SAV0(:, 1:j), 0);
    y = (T \ (U'*Sb));
    e = V0(:,1:j) * y ;
    x = x0 + e;
    r = b - A(x);

    normx = norm(x);
    normr = norm(r);
    normy = norm(y);
    normV0 = norm(SV0);

    res0(end+1) = norm(r);
    berr0(end+1) = normr/(normA*normx + normb);
    cnd0(end+1) = cond(SV0);
    cnd00(end+1) = cond(SAV0);
    sbnd0(end+1) = (normV0*normy)/normx;

end

% For restarted versions, we change the value of s
s = 2*(m+1);
hS = srhtb2(n, s); % sketching operator

%% sGMRES with truncated Arnoldi (restarted)
x0 = zeros(n,1);
r0 = b;

AV2 = [];
H2 = zeros(m+1,m);
SAV2 = [];
res2 = []; berr2 = [];	cnd2 = []; cnd22 = []; sbnd2 = [];

for restart = 1:nrestarts

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

    end

    x0 = x;
    r0 = r;

end

%% sGMRES with sketch-and-select Arnoldi
x0 = zeros(n,1);
r0 = b;

AV3 = [];
H3 = zeros(m+1,m);
SAV3 = [];
res3 = []; berr3 = []; cnd3 = []; cnd33 = []; sbnd3 = [];

for restart = 1:nrestarts

    V3 = r0/norm(r0);
    SV3 = hS(V3);
    Sb = hS(r0);

    for j = 1:m
        w = A(V3(:,j));
        AV3(:, j) = w;
        sw = hS(w); SAV3(:,j) = sw;
        [Q, R] = qr(SV3(:, 1:j), 0);
        coeffs = R \ (Q'*sw);
        [~,ind] = maxk(abs(coeffs),t);
        w = w - V3(:,ind)*coeffs(ind);
        H3(ind,j) = coeffs(ind);
        sw = hS(w);
        H3(j+1,j) = norm(sw);
        V3(:,j+1) = w/H3(j+1,j);
        SV3(:,j+1) = sw/H3(j+1,j);

        [U, T] = qr(SAV3(:, 1:j), 0);
        y = (T \ (U'*Sb));
        e = V3(:,1:j) * y ;
        x = x0 + e;
        r = b - A(x);

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV3 = norm(SV3);

        res3(end+1) = norm(r);
        berr3(end+1) = normr/(normA*normx + normb);
        cnd3(end+1) = cond(SV3);
        cnd33(end+1) = cond(SAV3);
        sbnd3(end+1) = (normV3*normy)/normx;

    end

    x0 = x;
    r0 = r;
end

% standard restarted GMRES
x0 = zeros(n,1);
r0 = b;
AV4 = [];
H4 = zeros(m+1,m);
res4 = [];  % stores residuals
berr4 = []; % stores backward error
cnd4 = [];  % stores conditon number of V
cnd44 = []; % stores condition number of A*V
sbnd4 = []; % stores sharper bound

for restart = 1:nrestarts

    V4 = r0/norm(r0);

    for j = 1:m
        w = A(V4(:,j));
        for i = 1:j
            for reo = 0:0
                h = V4(:,i)'*w;
                w = w - h*V4(:,i);
                H4(i,j) = h;
            end
        end
        H4(j+1,j) = norm(w);
        V4(:,j+1) = w/H4(j+1,j);

        y = H4(1:j+1,1:j)\(norm(r0)*eye(j+1,1));
        e = V4(:,1:j)*y;
        x = x0 + e;
        r = b - A(x);

        normx = norm(x);
        normr = norm(r);
        normy = norm(y);
        normV4 = norm(V4);

        res4(end+1) = normr;
        berr4(end+1) = normr/(normA*normx + normb);
        cnd4(end+1) = cond(V4);
        cnd44(end+1) = cond(A(V4));
        sbnd4(end+1) = (normV4*normy)/normx;

    end

    x0 = x;
    r0 = r;
end

figure
t = tiledlayout(2,2);

% Plot backward errors
nexttile
semilogy(berr1,'LineWidth',3); hold on;
semilogy(berr0, 'LineWidth',3);
semilogy(berr2, 'LineWidth',3);
semilogy(berr3,'LineWidth',3 );
semilogy(berr4, 'LineWidth',3);
xlabel("iteration",'interpreter','latex');
ylabel("backward error", 'interpreter','latex');
yticks([1e-15 1e-10 1e-5 1e-0])
ylim([1e-15 1e0]);
set(gca,'FontSize', 15, 'FontWeight', 'normal')

nexttile
semilogy(sbnd1,'LineWidth',3); hold on;
semilogy(sbnd0,'LineWidth',3);
semilogy(sbnd2,'LineWidth',3);
semilogy(sbnd3, 'LineWidth',3);
semilogy(sbnd4, 'LineWidth',3);
xlabel("iteration",'interpreter','latex');
yticks([1e-15, 1e-5, 1e5, 1e15 ])
ylim([1e-15 1e15])
ylabel("$\tau_{i}$ ",'interpreter','latex');
set(gca,'FontSize', 15, 'FontWeight', 'normal')

% plot condition numbers of V
nexttile
semilogy(cnd1,'LineWidth',3);  hold on;
semilogy(cnd0,'LineWidth',3);
semilogy(cnd2,'LineWidth',3);
semilogy(cnd3, 'LineWidth',3);
semilogy(cnd4,'LineWidth',3 );
xlabel("iteration",'interpreter','latex');
ylabel("$\kappa(B_{1:i})$",'interpreter','latex');
xlim([-1, m*nrestarts+1]);
ylim([1e-1 1e18]);
yticks([1e0, 1e5, 1e10, 1e15])
set(gca,'FontSize', 15, 'FontWeight', 'normal')

% plot condition numbers of A*V
nexttile
semilogy(cnd11,'LineWidth',3); hold on;
semilogy(cnd00,'LineWidth',3);
semilogy(cnd22,'LineWidth',3);
semilogy(cnd33, 'LineWidth',3);
semilogy(cnd44, 'LineWidth',3);
xlabel("iteration",'interpreter','latex');
ylabel("$\kappa(AB_{1:i})$",'interpreter','latex');
xlim([-1, m*nrestarts+1]);
ylim([1e-1 1e18]);
yticks([1e0, 1e5, 1e10, 1e15])

lgd = legend('sGMRES-trunc', 'sGMRES-saa', 'restarted sGMRES-trunc', ' restarted sGMRES-ssa','restarted GMRES', 'Location','southoutside');
lgd.Layout.Tile = "south";
lgd.NumColumns = 3;
set(gca,'FontSize', 15, 'FontWeight', 'normal')
set(gcf,'Units','pixels','Position',[300 10 800 800]);
