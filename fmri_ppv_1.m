function fmri_ppv_1

% This function visualizes various aspects of power, sample size, and
% positive predictive value calculation for single outcome measures.
%
% Copyright (C) Dirk Ostwald 2016
% -------------------------------------------------------------------------
clc
close all

%--------------------------------------------------------------------------
%                           Figure Properties
% -------------------------------------------------------------------------
% Default figure properties
set(0                                                       , ...
        'DefaultFigureColor'        , [1 1 1]               , ...
        'DefaultFigurePosition'     , get(0,'Screensize')   , ... 
        'DefaultTextInterpreter'    ,'Latex'                , ...
        'DefaultAxesFontSize'       , 16                 , ...
        'DefaultAxesFontName'       ,'Times New Roman'      , ...
        'DefaultAxesLineWidth'      , 1                     , ...               
        'DefaultLegendInterpreter'  , 'Latex'               , ...
        'DefaultLegendFontSize'     , 16                   , ...             
        'DefaultLegendBox'          , 'off'             )
        
% nice colors
cols    = get(groot,'DefaultAxesColorOrder'); 

% -------------------------------------------------------------------------
%                        Significance level and p-value
% -------------------------------------------------------------------------
t_min       = 1.5;                                                          % minimum T statistic
t_max       = 3.5;                                                          % maximum T statistic
t_res       = 1e3;                                                          % T statistic space resolution
t           = linspace(t_min, t_max, t_res);                                % T statistic space of interest
n           = 12;                                                           % sample size
alpha       = 0.05;                                                         % test size alpha
gamma_y     = 2.9;                                                          % test statistic
c_alpha     = icdf('T',1-alpha,n-1);                                        % critical value
R_alpha     = linspace(c_alpha,t_max,t_res);                                % critical region 
R_gamma     = linspace(gamma_y,t_max,t_res);                                % significance region 

% visualization
h = figure;
subplot(2,4,[1 2])
hold on
plot(t,         pdf('T',t,n-1)  , 'LineWidth', 1,'Color', 'k')
plot(c_alpha,   0               , 'ko', 'MarkerFaceColor', cols(1,:), 'MarkerSize', 20)
area(R_alpha,   pdf('T',R_alpha,n-1)  , 'FaceColor', cols(1,:), 'FaceAlpha', .2)
area(R_gamma,   pdf('T',R_gamma,n-1)  , 'FaceColor', cols(2,:), 'FaceAlpha', .5)
plot(gamma_y,   0               , 'ko', 'MarkerFaceColor', cols(2,:), 'MarkerSize', 20)
set(gca, 'xtick', [], 'xticklabel', {},'ytick', [], 'yticklabel', {})
annotation('textbox'  ,[.27 .90 .5 .0],'String','$\alpha$', 'LineStyle','none', 'Interpreter', 'Latex', 'FontSize', 50,'Color', cols(1,:));
annotation('textbox'  ,[.42 .90 .5 .0],'String','$\mbox{p}$', 'LineStyle','none', 'Interpreter', 'Latex', 'FontSize', 44,'Color', cols(2,:));
annotation(h, 'arrow', [.285 .285], [.81 .62], 'LineWidth', 1, 'Headwidth', 20, 'HeadLength', 20,'Color', cols(1,:));
annotation(h, 'arrow', [.43  .43] , [.81 .59], 'LineWidth', 1, 'Headwidth', 20, 'HeadLength', 20,'Color', cols(2,:));
annotation('textbox',[.17 .57 .5 .0],'String','$u_{\alpha}$', 'LineStyle','none', 'Interpreter', 'Latex', 'FontSize',  30);
annotation('textbox',[.35 .57 .5 .0],'String','$\gamma(Y^{\prime})$', 'LineStyle','none', 'Interpreter', 'Latex', 'FontSize', 30);
xlabel('$\gamma(Y)$','FontSize', 36)
text(-0.2, 1.2, 'A', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize',40)

% -------------------------------------------------------------------------
%                            Hypothesis Testing
% -------------------------------------------------------------------------
% test quality function
theta_0 = 0                                                                 ; % null hypothesis
theta_1 = 1                                                                 ; % alternative hypothesis
sigsqr  = 0.3                                                               ; % variance parameter
y       = linspace(-1,2,1e3)                                                ; % data space
c       = .6                                                                ; % critical value
qy      = linspace(c,2,1e3)                                                 ; % data space

% test function \phi
phi     = NaN(length(y));
for i = 1:length(y)
    if y(i) < c
        phi(i) = 0;
    else
        phi(i) = 1.4;
    end
end

% visualization
subplot(2,4,[3 4])
hold on
plot(y,  pdf('Normal',y,theta_0, sigsqr)   , 'LineWidth', 2,'Color', cols(1,:))
area(qy, pdf('Normal',qy,theta_0, sigsqr)  , 'FaceColor', cols(1,:), 'FaceAlpha', 1)
plot(y,  pdf('Normal',y,  theta_1, sigsqr) , 'LineWidth', 2, 'Color', cols(2,:))
area(qy, pdf('Normal',qy, theta_1, sigsqr) , 'FaceColor', cols(2,:), 'FaceAlpha',.5)
plot(y,  phi, 'k', 'LineWidth', 1)
leg = legend('$p_{\theta_0}(y)$', '$q(\theta_0)$','$p_{\theta_1}(y)$','$q(\theta_1)$','$\phi(y)$', 'Location', 'NorthWest');
set(leg, 'Interpreter', 'Latex', 'FontSize', 24)
set(gca, 'xtick', [theta_0, c, theta_1], 'xticklabel', {'\theta_0', 'c', '\theta_1'}, 'ytick', [], 'FontSize', 26)
text(-0.1, 1.2, 'B', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize',40)

% -------------------------------------------------------------------------
%                         Test Quality Function
% -------------------------------------------------------------------------
% H = 0: type I error probability as function of u and H = 1: power  
% as function of u
u_min   = -2;                                                               % minimum Gauss test threshold 
u_max   = 4;                                                                % maximum Gauss test threshold
u_res   = 1e2;                                                              % Gauss test threshold space resolution
u       = linspace(u_min,u_max,u_res);                                      % threshold space
theta_0 = 0;                                                                % null hypothesis expectation
theta_1 = 1;                                                                % alternative hypothesis expectation
sigsqr  = 1;                                                                % common variance

subplot(2,4,5)
plot(u, 1 - cdf('Normal', u, theta_0, sigsqr), 'LineWidth', 2)
xlabel('$u$', 'FontSize', 30)
ylabel('$p_{\theta_0}(\phi(Y) = 1)$', 'FontSize', 26)
title('$q_{\phi}(\theta_0)$', 'FontSize', 30)
set(gca, 'FontSize', 22)
xlim([u_min u_max])
box off
text(-0.5, 1.2, 'C', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize',40)

subplot(2,4,6)
plot(u, 1 - cdf('Normal', u, theta_1, sigsqr), 'LineWidth', 2)
xlabel('$u$', 'FontSize', 30)
ylabel('$p_{\theta_1}(\phi(Y) = 1)$', 'FontSize', 26)
title('$q_{\phi}(\theta_1)$', 'FontSize', 30)
set(gca, 'FontSize', 22)
box off
xlim([u_min u_max])


% power as function of theta_1 and variance
theta_1_min = 2;                                                            % H = 1 expectation minimum
theta_1_max = 8;                                                            % H = 1 expectation maximum
theta_1_res = 3e1;                                                          % H = 1 expectation space resolution
theta_1     = linspace(theta_1_min, theta_1_max, theta_1_res);              % H = 1 expectation space
sigsqr_min  = 1;                                                         % H = 0 & H = 1 variance minimum                                      
sigsqr_max  = 10;                                                            % H = 0 & H = 1 variance maximum
sigsqr_res  = 3e1;                                                          % H = 0 & H = 1 variance space resolution
sigsqr      = linspace(sigsqr_min, sigsqr_max, sigsqr_res);                 % H = 0 & H = 1 variance space
u           = 1.96;                                                         % test threshold
ps          = NaN(theta_1_res,sigsqr_res);                                  % power surface initialization

for i = 1:length(theta_1)
    for j = 1: length(sigsqr)
        ps(i,j) = 1 - cdf('Normal', u, theta_1(i), sigsqr(j));
    end
end

subplot(2,4,[7 8])
surf(sigsqr,theta_1, ps)
axis square
view([43 14])
xlim([sigsqr_min sigsqr_max])
ylim([theta_1_min theta_1_max])
zlim([.5 1])
xlabel('$\sigma^2$', 'FontSize',30)
ylabel('$\theta_1$', 'FontSize',30)
title('$1-\beta$', 'FontSize',32)
set(gca, 'FontSize', 22)
text(-0.8, 1.2, 'D', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize',40)

% position and print
pos = get(0,'Screensize');
pos(4) = 0.8*pos(4);
set(h, 'Position', pos)
print2pdf(fullfile(pwd,'Figures','fmri_power_gauss_test.pdf'),h,300)
print(h, '-djpeg', '-r300', fullfile(pwd,'Figures','fmri_power_gauss_test'))

% -------------------------------------------------------------------------
%                        Positive Predictive Value
% -------------------------------------------------------------------------
res             = 3e1;                                                      % space resolution
pi              = linspace(0,1,res);                                        % prior probability
alpha           = linspace(0,1,res);                                        % test size  p(\phi = 1|H = 0)
power           = linspace(0,1,res);                                        % test power p(\phi = 1|H = 1)

% initialize PPV surfaces
ppv             = NaN(res,res,3);

% cycle over first variable
for i = 1:res
    
    % cycle over second variable
    for j = 1:res
        
        % constant prior p(H=1) = 0.5
        ppv(i,j,1) = (power(i)*0.5)/(power(i)*0.5 + alpha(j)*(1-0.5));
        
        % constant size p(\phi = 1|H=0) = \alpha
        ppv(i,j,2) = (power(i)*pi(j))/(power(i)*pi(j) + 0.05*(1-pi(j)));

        % constant power p(\phi = 1|H=1) = 1 -\beta
        ppv(i,j,3) = (0.8*pi(i))/(0.8*pi(i) + alpha(j)*(1-pi(i)));

    end
end

h = figure;
subplot(1,3,1)
surf(alpha,power,ppv(:,:,1))
set(gca,'YDir','normal')
colorbar
xlabel('$\alpha$'   , 'FontSize', 26)
ylabel('$1-\beta$'  , 'FontSize', 26)
title('$\pi = 0.5$' , 'FontSize', 28)
set(gca, 'FontSize', 20)
view([45 20])
text(-0.2, 1.16, 'A', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize',30)


subplot(1,3,2)
surf(pi,power,ppv(:,:,2))
set(gca,'YDir','normal')
colorbar
xlabel('$\pi$', 'FontSize', 26)
ylabel('$1-\beta$', 'FontSize', 26)
title('$\alpha = 0.05$', 'FontSize', 28)
set(gca, 'FontSize', 20)
view([-45 20])
text(-0.2, 1.16, 'B', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize',30)

subplot(1,3,3)
surf(alpha,pi,ppv(:,:,3))
set(gca,'YDir','normal')
colorbar
xlabel('$\alpha$', 'FontSize', 26)
ylabel('$\pi$', 'FontSize', 26)
title('$1-\beta = 0.8$', 'FontSize', 28)
set(gca, 'FontSize', 20)
view([45 20])
text(-0.2, 1.16, 'C', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize',30)

% position and print
pos = get(0,'Screensize');
pos(4) = 0.35*pos(4);
set(h, 'Position', pos)
print2pdf(fullfile(pwd,'Figures','fmri_power_ppv.pdf'),h,300)
print(h, '-djpeg', '-r300', fullfile(pwd,'Figures','fmri_power_ppv'))

% -------------------------------------------------------------------------
%                   t- and non-central t-distributions
% -------------------------------------------------------------------------
T_min   = -4;                                                               % RV minimum
T_max   = 12;                                                               % RV maximum
T_res   = 1e3;                                                              % RV space resolution
T       = linspace(T_min,T_max,T_res);                                      % RV space
df      = [5 30];                                                           % degrees of freedom
delta   = 4;                                                                % non-centrality parameter

% visualization
h = figure;
subplot(2,2,[1 2])
hold on
plot(T,pdf('T',T,df(1)), 'LineWidth', 2, 'Color', cols(1,:))
plot(T,pdf('T',T,df(2)), 'LineWidth', 2, 'Color', cols(3,:))
plot(T,pdf('nct', T, df(1), delta(1)),'LineWidth', 2,'Color', cols(1,:), 'LineStyle', ':')
plot(T,pdf('nct', T, df(2), delta(1)),'LineWidth', 2,'Color', cols(3,:), 'LineStyle', ':')
leg = legend(   ['$f_{t(T_\nu; ' num2str(df(1)) ')}$'],...
                ['$f_{t(T_\nu; ' num2str(df(2)) ')}$'],...
                ['$f_{t(T_{\nu,\delta}; ' num2str(df(1)) ','  num2str(delta) ')}$'],...  
                ['$f_{t(T_{\nu,\delta}; ' num2str(df(2)) ','  num2str(delta) ')}$']);
set(leg, 'FontSize', 30);
xlabel('$x$', 'FontSize', 30)
set(gca, 'FontSize', 26);
text(-0.12, 1.2, 'A', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize',36)


% visualization
subplot(2,2,3)
hold on
plot(T,cdf('T',T,df(1)), 'LineWidth', 2, 'Color', cols(1,:))
plot(T,cdf('T',T,df(2)), 'LineWidth', 2, 'Color', cols(3,:))
xlim([T_min T_max])   
title('$\tilde{t}_\nu(x)$', 'FontSize', 30)
leg = legend(['$\nu = ' num2str(df(1)) '$'],['$\nu = ' num2str(df(2)) '$'], 'Location', 'SouthEast' );
set(leg, 'FontSize', 21);
xlabel('$x$', 'FontSize', 30)
set(gca, 'FontSize', 26);
text(-0.28, 1.15, 'B', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize',36)

subplot(2,2,4)
hold on
plot(T,cdf('nct',T,df(1),delta), 'LineWidth', 2, 'Color', cols(1,:),'LineStyle', ':')
plot(T,cdf('nct',T,df(2),delta), 'LineWidth', 2, 'Color', cols(3,:),'LineStyle', ':')
xlim([T_min T_max])   
title('$\tilde{t}_{\nu,\delta}(x)$', 'FontSize', 30)
leg = legend(['$\nu = ' num2str(df(1)) ', \delta = ' num2str(delta) '$'],...
             ['$\nu = ' num2str(df(2)) ', \delta = ' num2str(delta) '$'], 'Location', 'SouthEast');
set(leg, 'FontSize', 21);
xlabel('$x$', 'FontSize', 30)
set(gca, 'FontSize', 26);
text(-0.28, 1.15, 'C', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize',36)

% position and print
pos     = get(0,'Screensize');
pos(3)  = .6*pos(3);
set(h, 'Position', pos)
print2pdf(fullfile(pwd,'Figures','fmri_power_t_distributions.pdf'),h,300)
print(h, '-djpeg', '-r300', fullfile(pwd,'Figures','fmri_power_t_distributions'))

% -------------------------------------------------------------------------
%                One-sample t-test power and sample size functions
% -------------------------------------------------------------------------
alpha   = 0.05;                                                             % test size
pi      = [.5 .7];                                                          % prior p(H=1)
D_min   = 0;                                                                % minimum Cohen's D
D_max   = 1;                                                                % maximum Cohen's D
D_res   = 3e1;                                                              % Cohen's D space resolution
D       = linspace(D_min,D_max,D_res);                                      % Cohen's D parameter space 
n_min   = 2;                                                                % minimal sample size  
n_max   = 40;                                                               % maximum sample size
n       = n_min:n_max;                                                      % sample size space

% evaluate power surface
% -------------------------------------------------------------------------
ps      = NaN(length(n), length(D));                                        % power surface initialization

% cycle over sample size and Cohen's D
for i = 1:length(n)
    for j = 1:length(D)
        ps(i,j) = 1-cdf('nct',icdf('T',1-alpha,n(i)-1), n(i)-1,sqrt(n(i))*D(j));
    end
end

% visualization
h = figure;

% power surface
subplot(2,3,[1 4])
surf(D,n,ps);
ylabel('$n$', 'FontSize', 20);
xlabel('$D$', 'FontSize', 20);
zlabel('$1-\beta$', 'FontSize', 20);
text(-0.2, 1.1, 'A', 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize', 28)
view([-45 20])
axis square

% evaluate ppv surfaces
% -------------------------------------------------------------------------
ppv     = NaN(length(n), length(D),length(pi));                          % PPV surface initialization
lppv    = {'B', 'C'};                                                       % subplot labels

% cycle over prior,  sample size, and Cohen's D
for p = 1:length(pi)
    for i = 1:length(n)
        for j = 1:length(D)
            ppv(i,j,p)    = (ps(i,j)*pi(p))/(ps(i,j)*pi(p) + (1-pi(p))*alpha);
        end
    end
    
    % visualization
    subplot(2,3,1+p)
    surf(D,n,ppv(:,:,p));
    ylabel('$n$', 'FontSize', 20);
    xlabel('$D$', 'FontSize', 20);
    zlabel('$\mbox{PPV}$', 'FontSize', 20);
    text(-0.2, 1.15,lppv{p} , 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize', 28)
    axis square
    title(['$\pi$ = ' num2str(pi(p))])
    view([-45 20])
    zlim([0 1])
end

% evaluate minimal required sample sizes for a desired power
% -------------------------------------------------------------------------
alpha   = [0.05 0.001];                                                     % test size
dp_min  = 0.5;                                                              % minimum desired power
dp_max  = 0.9;                                                              % maximum desired power           
dp_res  = 3e1;                                                              % desired power space resolution
dp      = linspace(dp_min, dp_max, dp_res);                                 % desired power space
D_min   = .2;                                                               % minimum Cohen's D
D_max   = .5;                                                               % maximum Cohen's D
D_res   = 3e1;                                                              % Cohen's D space resolution
D       = linspace(D_min,D_max,D_res);                                      % Cohen's D parameter space 
ns      = NaN(length(dp), length(D), length(alpha));                        % required sample size surface initialization
lns     = {'D', 'E'};                                                       % subplot labels

% cycle over test size, desired sample size, and Cohen's D
for a = 1:length(alpha)
    for i = 1:length(dp)
        for j = 1:length(D)

            % sample size search initialization
            ni      = 2;                                                                % sample size iterand initialization
            power   = 1-cdf('nct',icdf('T',1-alpha(a),ni-1), ni-1,sqrt(ni)*D(j));       % minimum power considered

            % increase sample size in steps of one, until desired power reached
            while power <= dp(i)
                ni      = ni+1;                                                         % increase iterand sample size
                power   = 1-cdf('nct',icdf('T',1-alpha(a),ni-1), ni-1,sqrt(ni)*D(j));   % evaluate power
            end

            % minimum sample size to achieve power dp(i) at non-centrality D(j) and test size alpha(a)
            ns(i,j,a)   = ni;
        end
    end
    
    % visualization
    subplot(2,3,4+a)
    surf(D,dp,ns(:,:,a));
    ylabel('$1-\beta$', 'FontSize', 20);
    xlabel('$D$', 'FontSize', 20);
    zlab = zlabel('$n$', 'FontSize', 20, 'Rotation', 0);
    text(-0.2, 1.15, lns{a}, 'Units','Normalized', 'VerticalAlignment', 'Top', 'FontSize', 28)
    axis square
    title(['$\alpha$ = ' num2str(alpha(a))])
    view([45 20])
    xlim([D_min D_max])
    ylim([dp_min dp_max])
 
end

% maximize nicely
set(h, 'Position', get(0,'Screensize'))

% print figure 
print2pdf(fullfile(pwd,'Figures','fmri_power_one_sample_t_test.pdf'),h,100)
print(h, '-djpeg', '-r300', fullfile(pwd,'Figures','fmri_power_one_sample_t_test'))


end

% -------------------------------------------------------------------------
%                           Subfunctions
% -------------------------------------------------------------------------

function print2pdf(pdfFileName,handle,dpi)

% If no handle is provided, use the current figure as default
if nargin<1
    [fileName,pathName] = uiputfile('*.pdf','Save to PDF file:');
    if fileName == 0; return; end
    pdfFileName = [pathName,fileName];
end
if nargin<2
    handle = gcf;
end
if nargin<3
    dpi = 150;
end

% Backup previous settings
prePaperType        = get(handle,'PaperType');
prePaperUnits       = get(handle,'PaperUnits');
preUnits            = get(handle,'Units');
prePaperPosition    = get(handle,'PaperPosition');
prePaperSize        = get(handle,'PaperSize');

% Make changing paper type possible
set(handle,'PaperType','<custom>');

% Set units to all be the same
set(handle,'PaperUnits','inches');
set(handle,'Units','inches');

% Set the page size and position to match the figure's dimensions
paperPosition       = get(handle,'PaperPosition');
position            = get(handle,'Position');
set(handle,'PaperPosition',[0,0,position(3:4)]);
set(handle,'PaperSize',position(3:4));

% Save the pdf (this is the same method used by "saveas")
print(handle,'-dpdf', pdfFileName,sprintf('-r%d',dpi))

% Restore the previous settings
set(handle,'PaperType',prePaperType);
set(handle,'PaperUnits',prePaperUnits);
set(handle,'Units',preUnits);
set(handle,'PaperPosition',prePaperPosition);
set(handle,'PaperSize',prePaperSize);
end

