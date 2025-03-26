% test1_torso
% Plots the backward error curves for sGMRES applied to the torso matrix.

clear all
close all
clc

% load matrix
load('torso3.mat');
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

nrestarts = 1; % max number of restarts
m = 300;       % number of Arnoldi iterations
t = 2;         % Arnoldi truncation parameter

s = 2*(m+1);
hS = srhtb2(n, s); % sketching operator

%% sGMRES with truncated Arnoldi
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

%% standard restarted GMRES
x0 = zeros(n,1);
r0 = b;
AV4 = [];
H4 = zeros(m+1,m);
res4 = [];  % stores residuals
berr4 = []; % stores backward error
cnd4 = [];  % stores conditon number of V
cnd44 = []; % stores condition number of A*V
sbnd4 = []; % stores sharper backward error bound

for restart = 1:nrestarts

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

        res4(end+1) = normr;
        berr4(end+1) = normr/(normA*normx + normb);
        cnd4(end+1) = cond(V4);
        cnd44(end+1) = cond(AV4);
        sbnd4(end+1) = (normV4*normy)/normx;

    end

    x0 = x;
    r0 = r;
end


figure
t = tiledlayout(2,2);

% Plot backward errors
nexttile
semilogy(berr2, 'LineWidth',3); hold on;
semilogy(berr3,'LineWidth',3 );
semilogy(berr4, 'LineWidth',3);
xlabel("iteration",'interpreter','latex');
ylabel("backward error", 'interpreter','latex');
yticks([1e-15 1e-10 1e-5 1e0])
ylim([1e-15 1e0]);
set(gca,'FontSize', 15, 'FontWeight', 'normal')

nexttile
semilogy(sbnd2,'LineWidth',3); hold on;
semilogy(sbnd3, 'LineWidth',3);
semilogy(sbnd4, 'LineWidth',3);
xlabel("iteration",'interpreter','latex');
ylim([1e0 1e15]);
yticks([1e0, 1e5, 1e10, 1e15])
ylabel("$\tau_{i}$",'interpreter','latex');
set(gca,'FontSize', 15, 'FontWeight', 'normal')

% plot condition numbers of V
nexttile
semilogy(cnd2,'LineWidth',3); hold on;
semilogy(cnd3, 'LineWidth',3);
semilogy(cnd4,'LineWidth',3 );
xlabel("iteration",'interpreter','latex');
ylabel("$\kappa(B_{1:i})$",'interpreter','latex');
ylim([1e-1 1e18]);
yticks([1e0, 1e5, 1e10, 1e15])
set(gca,'FontSize', 15, 'FontWeight', 'normal')

% plot condition numbers of A*V
nexttile
semilogy(cnd22,'LineWidth',3); hold on;
semilogy(cnd33, 'LineWidth',3);
semilogy(cnd44, 'LineWidth',3);
xlabel("iteration",'interpreter','latex');
ylabel("$\kappa(AB_{1:i})$",'interpreter','latex');
ylim([1e-1 1e18]);
yticks([1e0, 1e5, 1e10, 1e15])

lgd = legend('sGMRES-trunc', 'sGMRES-ssa','GMRES', 'Location','southoutside');
lgd.Layout.Tile = "south";
lgd.NumColumns = 5;
set(gca,'FontSize', 15, 'FontWeight', 'normal')
set(gcf,'Units','pixels','Position',[300 10 800 800]);
