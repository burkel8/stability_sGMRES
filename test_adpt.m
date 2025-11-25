% load matrix
load('fs_760_1.mat');
% load('sherman2.mat');
A = Problem.A;

nzeros = nonzeros(A);
normA = norm(nzeros);

n = size(A,1);

% seed random number generator to reproduce results
rng(1);

% construct right hand side
b = randn(n,1);
b = b/norm(b);
normb = norm(b);
x0 = zeros(n,1);

maxiters = 1e3; % max number of iterations
m = 50;         % number of Arnoldi iterations per cycle：50/100
nrestarts = ceil(maxiters/m); % max number of restarts
t = 1;          % Arnoldi truncation parameter
tol = 1e-15;    % convergence tolerance
toltau = eps;
M = @(x) x;     % No preconditioner

[x, res, berr, cndV, cndAV, sbnd, taulog,  tlog] = gmres_restart(A, b, x0, M, tol, m, nrestarts, m);
numflops = compute_flops(tlog, n);
[x2, res2, berr2, cndV2, cndAV2, sbnd2, taulog2, tlog2] = sgmres_truncatedArnoldi_restart(A, b, x0, M, tol, m, maxiters, t);
numflops2 = compute_flops(tlog2, n);
[x3, res3, berr3, cndV3, cndAV3, sbnd3, taulog3, tlog3] = sgmres_truncatedArnoldi_adpt(A, b, x0, M, tol, m, maxiters, t, toltau);
numflops3 = compute_flops(tlog3, n);

figure
t = tiledlayout(2,3);
% t = tiledlayout(3,3);

nexttile
semilogy(berr,'LineWidth', 3); hold on;
semilogy(berr2, 'LineWidth', 3); hold on;
semilogy(berr3, 'LineWidth', 3, 'LineStyle', ':');

title("$m = 50$",'interpreter','latex')
xlabel("iteration",'interpreter','latex');
ylabel("backward error", 'interpreter','latex');
xlim([0 maxiters])
ylim([1e-15 1e0])
yticks([1e-15 1e-10 1e-5 1e0]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

nexttile
plot(tlog,'LineWidth',3); hold on;
plot(tlog2, 'LineWidth', 3); hold on;
plot(tlog3, 'LineWidth', 3, 'LineStyle', ':');

title("$m = 50$",'interpreter','latex')
xlabel("iteration",'interpreter','latex');
ylabel("$\bar{t}$", 'interpreter','latex');
xlim([0 maxiters])
ylim([-0.1*m, m+0.1*m])
% yticks([0 25 50]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

nexttile
semilogy(taulog,'LineWidth',3); hold on;
semilogy(taulog2, 'LineWidth', 3); hold on;
semilogy(taulog3, 'LineWidth', 3, 'LineStyle', ':');

title("$m = 50$",'interpreter','latex')
xlabel("iteration",'interpreter','latex');
ylabel("$\tau$", 'interpreter','latex');
xlim([0 maxiters])
ylim([1e-1, 1e16])
yticks([1e0 1e5 1e10 1e15]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

%% Use different m.
m = 100;        % number of Arnoldi iterations per cycle
nrestarts = ceil(maxiters/m); % max number of restarts
t = 1;

[x, res, berr, cndV, cndAV, sbnd, taulog,  tlog] = gmres_restart(A, b, x0, M, tol, m, nrestarts, m);
numflopsm = compute_flops(tlog, n);
[x2, res2, berr2, cndV2, cndAV2, sbnd2, taulog2, tlog2] = sgmres_truncatedArnoldi_restart(A, b, x0, M, tol, m, maxiters, t);
numflopsm2 = compute_flops(tlog2, n);
[x3, res3, berr3, cndV3, cndAV3, sbnd3, taulog3, tlog3] = sgmres_truncatedArnoldi_adpt(A, b, x0, M, tol, m, maxiters, t, toltau);
numflopsm3 = compute_flops(tlog3, n);

nexttile
semilogy(berr,'LineWidth', 3); hold on;
semilogy(berr2, 'LineWidth', 3); hold on;
semilogy(berr3, 'LineWidth', 3, 'LineStyle', ':');

title("$m = 100$",'interpreter','latex')
xlabel("iteration",'interpreter','latex');
ylabel("backward error", 'interpreter','latex');
xlim([0 maxiters])
ylim([1e-15 1e0])
yticks([1e-15 1e-10 1e-5 1e0]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

nexttile
plot(tlog,'LineWidth',3); hold on;
plot(tlog2, 'LineWidth', 3); hold on;
plot(tlog3, 'LineWidth', 3, 'LineStyle', ':');

title("$m = 100$",'interpreter','latex')
xlabel("iteration",'interpreter','latex');
ylabel("$\bar{t}$", 'interpreter','latex');
xlim([0 maxiters])
ylim([-0.1*m, m+0.1*m])
% yticks([0 50 100]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')

nexttile
semilogy(taulog,'LineWidth',3); hold on;
semilogy(taulog2, 'LineWidth', 3); hold on;
semilogy(taulog3, 'LineWidth', 3, 'LineStyle', ':');

title("$m = 100$",'interpreter','latex')
xlabel("iteration",'interpreter','latex');
ylabel("$\tau$", 'interpreter','latex');
xlim([0 maxiters])
ylim([1e-1, 1e16])
yticks([1e0 1e5 1e10 1e15]);
set(gca,'FontSize', 20, 'FontWeight', 'normal')


% %% Use different m.
% m = 400;        % number of Arnoldi iterations per cycle：50/100
% nrestarts = ceil(maxiters/m); % max number of restarts
% t = 1;
% 
% [x, res, berr, cndV, cndAV, sbnd, taulog,  tlog] = gmres_restart(A, b, x0, M, tol, m, nrestarts, m);
% numflopsm = compute_flops(tlog, n);
% [x2, res2, berr2, cndV2, cndAV2, sbnd2, taulog2, tlog2] = sgmres_truncatedArnoldi_restart(A, b, x0, M, tol, m, maxiters, t);
% numflopsm2 = compute_flops(tlog2, n);
% [x3, res3, berr3, cndV3, cndAV3, sbnd3, taulog3, tlog3] = sgmres_truncatedArnoldi_adpt(A, b, x0, M, tol, m, maxiters, t, toltau);
% numflopsm3 = compute_flops(tlog3, n);
% 
% nexttile
% semilogy(berr,'LineWidth', 3); hold on;
% semilogy(berr2, 'LineWidth', 3); hold on;
% semilogy(berr3, 'LineWidth', 3, 'LineStyle', ':');
% 
% title("$m = 400$",'interpreter','latex')
% xlabel("iteration",'interpreter','latex');
% ylabel("backward error", 'interpreter','latex');
% xlim([0 maxiters])
% ylim([1e-15 1e0])
% yticks([1e-15 1e-10 1e-5 1e0]);
% set(gca,'FontSize', 20, 'FontWeight', 'normal')
% 
% nexttile
% plot(tlog,'LineWidth',3); hold on;
% plot(tlog2, 'LineWidth', 3); hold on;
% plot(tlog3, 'LineWidth', 3, 'LineStyle', ':');
% 
% title("$m = 400$",'interpreter','latex')
% xlabel("iteration",'interpreter','latex');
% ylabel("$\bar{t}$", 'interpreter','latex');
% xlim([0 maxiters])
% ylim([-0.1*m, m+0.1*m])
% % yticks([0 50 100]);
% set(gca,'FontSize', 20, 'FontWeight', 'normal')
% 
% nexttile
% semilogy(taulog,'LineWidth',3); hold on;
% semilogy(taulog2, 'LineWidth', 3); hold on;
% semilogy(taulog3, 'LineWidth', 3, 'LineStyle', ':');
% 
% title("$m = 400$",'interpreter','latex')
% xlabel("iteration",'interpreter','latex');
% ylabel("$\tau$", 'interpreter','latex');
% xlim([0 maxiters])
% ylim([1e-1, 1e16])
% yticks([1e0 1e5 1e10 1e15]);
% set(gca,'FontSize', 20, 'FontWeight', 'normal')

lgd = legend('restarted GMRES', 'restarted sGMRES', 'restarted sGMRES with adapt $t$', 'Location','southoutside','interpreter','latex');
lgd.Layout.Tile = "south";
lgd.NumColumns = 3;
set(gca,'FontSize', 20, 'FontWeight', 'normal')
set(gcf,'Units','pixels','Position',[100 100 1000 500]);



