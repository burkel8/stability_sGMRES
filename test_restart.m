load('stomach.mat');
A = Problem.A;
nzeros = nonzeros(A);
normA = norm(nzeros);
n = size(A,1);
rng(1);
b = randn(n,1);

% Test
normb = norm(b);
x0 = zeros(n,1);

maxiters = 750;% max number of iterations
m = 150;       % number of Arnoldi iterations per cycle
nrestarts = maxiters/m; % max number of restarts
t = 3;         % Arnoldi truncation parameter
tol = 1e-15;   % convergence tolerance
M = @(x) x;    % No preconditioner

[x, res, berr, cndV, cndAV, sbnd, taulog,  ~] = gmres_restart(A, b, x0, M, tol, m, nrestarts, m);
[x1, res1, berr1, cndV1, cndAV1, sbnd1, taulog1, ~] = sgmres_ssa_restart(A, b, x0, tol, m, maxiters, t);
[x2, res2, berr2, cndV2, cndAV2, sbnd2, taulog2, ~] = sgmres_truncatedArnoldi_restart(A, b, x0, M, tol, m, maxiters, t); %

m = 750;
nrestarts = maxiters/m;
[x3, res3, berr3, cndV3, cndAV3, sbnd3, taulog3, ~] = sgmres_ssa_restart(A, b, x0, tol, m, maxiters, t);
[x4, res4, berr4, cndV4, cndAV4, sbnd4, taulog4, ~] = sgmres_truncatedArnoldi_restart(A, b, x0, M, tol, m, maxiters, t); %


figure
t = tiledlayout(2,2);

nexttile
semilogy(berr4,'LineWidth', 3, 'Marker', '+'); hold on;
semilogy(berr3,'LineWidth', 3); hold on;
semilogy(berr2,'LineWidth', 3); hold on;
semilogy(berr1, 'LineWidth',3, 'LineStyle','--'); hold on
semilogy(berr, 'LineWidth',3); hold on
xlabel("iteration",'interpreter','latex');
ylabel("backward error", 'interpreter','latex');
xlim([0 800])
ylim([1e-15 1e0])
yticks([1e-15 1e-10 1e-5 1e0]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

nexttile
semilogy(taulog4,'LineWidth', 3, 'Marker', '+'); hold on;
semilogy(taulog3,'LineWidth', 3); hold on;
semilogy(taulog2,'LineWidth',3); hold on;
semilogy(taulog1, 'LineWidth', 3, 'LineStyle','--'); hold on;
semilogy(taulog, 'LineWidth', 3); hold on
xlabel("iteration",'interpreter','latex');
ylabel("$\tau_i$", 'interpreter','latex');
xlim([0 800])
ylim([1e-1, 1e20])
yticks([1e0 1e5 1e10 1e15 1e20]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

nexttile
semilogy(cndV4,'LineWidth', 3, 'Marker', '+'); hold on;
semilogy(cndV3,'LineWidth', 3); hold on;
semilogy(cndV2,'LineWidth',3); hold on;
semilogy(cndV1, 'LineWidth', 3, 'LineStyle','--'); hold on;
semilogy(cndV, 'LineWidth', 3); hold on
xlabel("iteration",'interpreter','latex');
ylabel("$\kappa(B_{1:i})$", 'interpreter','latex');
xlim([0 800])
ylim([1e-1, 1e20])
yticks([1e0 1e5 1e10 1e15 1e20]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

nexttile
semilogy(cndAV4,'LineWidth', 3, 'Marker', '+'); hold on;
semilogy(cndAV3,'LineWidth', 3); hold on;
semilogy(cndAV2,'LineWidth',3); hold on;
semilogy(cndAV1, 'LineWidth', 3, 'LineStyle','--'); hold on;
semilogy(cndAV, 'LineWidth', 3); hold on
xlabel("iteration",'interpreter','latex');
ylabel("$\kappa(AB_{1:i})$", 'interpreter','latex');
xlim([0 800])
ylim([1e-1, 1e20])
yticks([1e0 1e5 1e10 1e15 1e20]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

lgd = legend('sGMRES-trunc', 'sGMRES-ssa', 'restarted sGMRES-trunc', 'restarted sGMRES-ssa', 'restarted GMRES', 'Location','southoutside','interpreter','latex');
lgd.Layout.Tile = "south";
lgd.NumColumns = 3;
set(gca,'FontSize', 20, 'FontWeight', 'normal')
set(gcf,'Units','pixels','Position',[100 100 900 600]);