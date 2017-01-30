<<<<<<< HEAD:common/plot/FormatPlot.m
function [ ] = FormatPlot( h, label_x,label_y,font_size)
%SAVEPLOT Summary of this function goes here
%   Detailed explanation goes here

%font_size = 20;
set(gca,'FontSize',font_size);
xlabel(label_x,'FontSize',font_size);
ylabel(label_y,'FontSize',font_size);
%title(title,'FontSize',font_size);
set(gcf,'color','w');
axis square; axis tight;

end

=======
function [ output_args ] = FormatPlot( h, label_x,label_y,title,file_name)
%SAVEPLOT Summary of this function goes here
%   Detailed explanation goes here

font_size = 20;
set(gca,'FontSize',font_size);
xlabel(label_x,'FontSize',font_size);
ylabel(label_y,'FontSize',font_size);
set(gcf,'color','w');
axis square; axis tight;

% if (~isempty(file_name)) 
%     export_fig(['figures/' file_name],'-m4','-transparent');
% end
end

>>>>>>> 7a565c5bdd8e37cc6cc93a81f492726ba9c9ea2b:Chapter7/FormatPlot.m
