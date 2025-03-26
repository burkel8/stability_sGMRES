% test_random.m
% Plots the backward error curves for sGMRES applied to a random matrix.

clear all
close all
clc

rng(1);
n = 400;
A = gallery('randsvd', [n, n], 1e1, 1);
[U, Sigma, V] = svd(A);
b = V(:, 3);
nzeros = nonzeros(A);
normA = norm(nzeros);

A = @(x) A*x;

normb = norm(b);

m = n;
t = 2;
s = min(n, 2*(m*+1));

hS = @(x) x; % S is the identity matrix

%% sGMRES with truncated Arnoldi
x0 = zeros(n, 1);
r0 = b;

AV1 = [];
H1 = zeros(m+1,m);
SAV1 = [];
res1 = []; berr1 = [];	cnd1 = []; cnd11 = []; sbnd1 = [];

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
    normV1 = norm(V1);

    res1(end+1) = normr;
    berr1(end+1) = normr/(normA*normx + normb);
    cnd1(end+1) = cond(V1);
    cnd11(end+1) = cond(AV1);
    sbnd1(end+1) = (normV1*normy)/normx;

end


%% sGMRES with sketch-and-select Arnoldi
x0 = zeros(n,1);
r0 = b;

AV0 = [];
H0 = zeros(m+1,m);
SAV0 = [];
res0 = []; berr0 = []; cnd0 = []; cnd00 = []; sbnd0 = [];

V0 = r0/norm(r0);
SV0 = hS(V0);
Sb = hS(r0);

for j = 1:m
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


%% standard Arnoldi
x0 = zeros(n, 1);
r0 = b;
AV4 = [];
H4 = zeros(m+1,m);
res4 = [];  % stores residuals
berr4 = []; % stores backward error
cnd4 = [];  % stores conditon number of V
cnd44 = []; % stores condition number of AV
sbnd4 = [];


V4 = r0/norm(r0);

for j = 1:m
    w = A(V4(:,j));
    AV4(:, j) = w;
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

    res4(end+1) = norm(r);
    berr4(end+1) = norm(r)/(normA*norm(x) + normb);
    cnd4(end+1) = cond(V4);
    cnd44(end+1) = cond(AV4);
    sbnd4(end+1) = (normV4*normy)/normx;
end

figure
t = tiledlayout(2,2);

nexttile
semilogy(berr1,'LineWidth',3); hold on;
semilogy(berr4, 'LineWidth',3,'Color','#EDB120');
semilogy(berr0, 'LineWidth',3,'Color','#D95319');
xlabel("iteration",'interpreter','latex');
ylabel("backward error", 'interpreter','latex');
yticks([1e-15 1e-10 1e-5 1e0])
set(gca,'FontSize', 15, 'FontWeight', 'normal')

nexttile
semilogy(sbnd1,'LineWidth',3); hold on;
semilogy(sbnd0,'LineWidth',3);
semilogy(sbnd4, 'LineWidth',3);
xlabel("iteration",'interpreter','latex');
ylim([1e0,1e20])
yticks([1e0 1e5 1e10 1e15 1e20 ])
ylabel("$\tau_{i}$",'interpreter','latex');
set(gca,'FontSize', 15, 'FontWeight', 'normal')

nexttile
semilogy(cnd1,'LineWidth',3);  hold on;
semilogy(cnd0,'LineWidth',3);
semilogy(cnd4,'LineWidth',3 );
xlabel("iteration",'interpreter','latex');
yticks([1e0 1e5 1e10 1e15 1e20 ])
ylabel("$\kappa(B_{1:i})$",'interpreter','latex');
set(gca,'FontSize', 15, 'FontWeight', 'normal')

nexttile
semilogy(cnd11,'LineWidth',3); hold on;
semilogy(cnd00,'LineWidth',3);
semilogy(cnd44, 'LineWidth',3);

xlabel("iteration",'interpreter','latex');
yticks([1e0 1e5 1e10 1e15 1e20 ])
ylabel("$\kappa(AB_{1:i})$",'interpreter','latex');
set(gca,'FontSize', 15, 'FontWeight', 'normal')

set(gca,'FontSize', 15, 'FontWeight', 'normal')
lgd = legend('sGMRES-trunc', 'sGMRES-saa ','GMRES','Location','southoutside');
lgd.Layout.Tile = "south";
lgd.NumColumns = 5;
set(gcf,'Units','pixels','Position',[300 100 800 800]);

