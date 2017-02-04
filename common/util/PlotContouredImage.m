function [ h ] = PlotContouredImage( image,map_size )
%PLOTCONTOUREDIMAGE Summary of this function goes here
%   Detailed explanation goes here

x = linspace(1,map_size(2),map_size(2));
y = linspace(1,map_size(1),map_size(1));
[xx,yy]=meshgrid(x,y);


min_sat = min(image(:));
max_sat = max(image(:));



h=pcolor(xx,yy,flipud(image)); shading interp

% hold on;
% [~,hfigc] = contour(xx, yy, flipud(image),...
%     linspace(min_sat,max_sat,3));
% set(hfigc, ...
%     'LineWidth',1.0, ...
%     'Color', [1 1 1]);

end

