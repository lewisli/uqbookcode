% NonParametric.m
% Generates illustration and figures for non-parametric density estimation
% in Chapter 7.2.5
% Author: Lewis Li
% Original Date: Nov 22nd 2016


% Generate some fake data for illustration. On purpose i will use a
% Gaussian Mixture model
num_mixtures = 10;
num_dim = 2;
rng(1); % For reproducibility
mu = zeros(num_mixtures,num_dim);
sigma = zeros(num_dim,num_dim,num_mixtures);

% Populate each Gaussian with random mean and covariance
for i = 1:num_mixtures
    mu(i,:) = randn(num_dim,1)*5;
    A = rand(num_dim);
    A = 0.5*(A+A');
    sigma(:,:,i) = A + num_dim*eye(num_dim);
end

% Uniform mixture weights
p = ones(1,num_mixtures)/num_mixtures;

% Create GM object and sample from it
obj = gmdistribution(mu,sigma,p);
Y = random(obj,1000);
h = Y(:,1);
d = Y(:,2);
x_range = [min(d) max(d)];
y_range = [min(h) max(h)];

hold on;
h1 = scatter(d,h,50,'filled');
dobs = -1.3;
plot([dobs dobs],y_range,'r','LineWidth',3)
text(dobs+0.25,y_range(1)*0.85,'d_{obs}','FontSize',20)

grid on;
ax = gca;
FormatPlot(h1,'d','h','','kde_raw.png');
%%
% generate a Gaussian mixture with distant modes
addpath('../common/kde2d');
addpath('../common/export_fig/');
MIN_XY = [min(d) min(h)];
MAX_XY = [max(d) max(h)];
[bandwidth,density,xcoord,ycoord]=kde2d(Y,512,MIN_XY,MAX_XY);

%%
figure
subplot(1,2,1);
h2=surf(xcoord,ycoord,density,'LineStyle','none'), view([0,90]);
grid on;
hold on;
plot3([dobs dobs],y_range,[1 1],'r','LineWidth',3)
text(dobs+0.25,y_range(1)*0.85,1,'d_{obs}','FontSize',20,'color','r')
FormatPlot(h2,'d','h','','kde_2d.png');

subplot(1,2,2);
[c,cindex] = min(abs(xcoord(1,:)-dobs))
conditional = density(:,cindex);
forecast = ycoord(:,cindex);

Area=trapz(forecast,conditional);
%figure;
h3=plot(forecast,conditional./Area,'LineWidth',2);
FormatPlot(h3,'h','f(h|d_{obs})','','kde_conditional.png');