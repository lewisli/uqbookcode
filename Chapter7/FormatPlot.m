function [ output_args ] = FormatPlot( h, label_x,label_y,title,file_name)
%SAVEPLOT Summary of this function goes here
%   Detailed explanation goes here

font_size = 20;
set(gca,'FontSize',font_size);
xlabel(label_x,'FontSize',font_size);
ylabel(label_y,'FontSize',font_size);
set(gcf,'color','w');
axis square; axis tight;


export_fig(['figures/' file_name],'-m4','-transparent');
end

