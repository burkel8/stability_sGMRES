load('torso3.mat');
A = Problem.A;
nzeros = nonzeros(A);
normA = norm(nzeros);
n = size(A,1);
rng(1);
b = randn(n,1);

% Test
normb = norm(b);
x0 = zeros(n,1);

maxiters = 300;  % max number of iterations
m = 300;         % number of Arnoldi iterations per cycle：50/100
nrestarts = ceil(maxiters/m); % max number of restarts
t = 2;         % Arnoldi truncation parameter
tol = 1e-15;   % convergence tolerance
M = @(x) x;    % No preconditioner

[x, res, berr, cndV, cndAV, sbnd, taulog,  ~] = gmres_restart(A, b, x0, M, tol, m, nrestarts, m);
[x1, res1, berr1, cndV1, cndAV1, sbnd1, taulog1, ~] = sgmres_ssa_restart(A, b, x0, tol, m, maxiters, t);
[x2, res2, berr2, cndV2, cndAV2, sbnd2, taulog2, ~] = sgmres_truncatedArnoldi_restart(A, b, x0, M, tol, m, maxiters, t); %


figure
t = tiledlayout(2,2);

nexttile
semilogy(berr2,'LineWidth', 3); hold on;
semilogy(berr, 'LineWidth',3,'Color','#EDB120'); hold on
semilogy(berr1, 'LineWidth',3,'Color','#D95319', 'LineStyle','--');
xlabel("iteration",'interpreter','latex');
ylabel("backward error", 'interpreter','latex');
xlim([0 m+40])
ylim([1e-15 1e0])
yticks([1e-15 1e-10 1e-5 1e0]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

nexttile
semilogy(taulog2,'LineWidth',3); hold on;
semilogy(taulog1, 'LineWidth', 3, 'LineStyle','--'); hold on;
semilogy(taulog, 'LineWidth', 3);
xlabel("iteration",'interpreter','latex');
ylabel("$\tau_i$", 'interpreter','latex');
xlim([0 m+40])
ylim([1e-1, 1e16])
yticks([1e0 1e5 1e10 1e15]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

nexttile
semilogy(cndV2,'LineWidth',3); hold on;
semilogy(cndV1, 'LineWidth', 3, 'LineStyle','--'); hold on;
semilogy(cndV, 'LineWidth', 3);
xlabel("iteration",'interpreter','latex');
ylabel("$\kappa(B_{1:i})$", 'interpreter','latex');
xlim([0 m+40])
ylim([1e-1, 1e16])
yticks([1e0 1e5 1e10 1e15]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

nexttile
semilogy(cndAV2,'LineWidth',3); hold on;
semilogy(cndAV1, 'LineWidth', 3, 'LineStyle','--'); hold on;
semilogy(cndAV, 'LineWidth', 3);
xlabel("iteration",'interpreter','latex');
ylabel("$\kappa(AB_{1:i})$", 'interpreter','latex');
xlim([0 m+40])
ylim([1e-1, 1e25])
yticks([1e0 1e5 1e10 1e15 1e20 1e25]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

lgd = legend('sGMRES-trunc', 'sGMRES-ssa', 'GMRES', 'Location','southoutside','interpreter','latex');
lgd.Layout.Tile = "south";
lgd.NumColumns = 3;
set(gca,'FontSize', 20, 'FontWeight', 'normal')
set(gcf,'Units','pixels','Position',[100 100 900 600]);
