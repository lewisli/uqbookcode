% Section71.m
% Generates illustration and figures for Chapter 7.1
% Author: Lewis Li
% Original Date: Nov 22nd 2016

% Generate some data
rng(1); % For reproducibility
num_samples = 500;

prior_salaries = sort((randn(num_samples,1)+4)*10000+20000);
apartment_sizes = max(100,prior_salaries*0.01 + randn(num_samples,1)*100);

x = prior_salaries;
y = apartment_sizes;

fitresult = fit(x,y,'poly1');
p11 = predint(fitresult,x,0.95,'observation','off');

fitted_line = x*fitresult.p1 + fitresult.p2;



x_range = [min(prior_salaries) max(prior_salaries)];
y_range = [min(apartment_sizes)*0.5 max(apartment_sizes)];

dobs = 50000;

forecast_val = dobs*fitresult.p1 + fitresult.p2;

hold on, 
h_scatter = scatter(x,y,'filled');
h_interval = plot(x,p11,'k:','LineWidth',3');
h_fit = plot(x,fitted_line,'k','LineWidth',3);
plot([dobs dobs],y_range,'r','LineWidth',3)
plot([x_range(1) dobs],[forecast_val forecast_val],'g','LineWidth',3)

text(dobs*1.025,y_range(1)*1.5,'d_{obs}','FontSize',20)
legend([h_scatter(1), h_fit, h_interval(1)],'Prior Samples',...
    'Best Fit','Prediction Interval','location','se');
FormatPlot(h1,'Salary ($)','Apartment Size (Sq Ft)','','');
