clc
clear

wave = ["[1.6]","[0.555]"];
angles =  ["60.0","30.0","15.0","0.0","-15.0","-60.0"];
name = "50_50_angles_and_waves_";
ch = [1 ,1, 2,1,2, 1];
%%

        figure (1)
        loss = zeros(50,50);
        
        for j = 1:size(angles,2)
           load(name+wave(ch(j))+"_"+angles(j)+".mat");
           loss = loss + a_dict.loss;
        end
        subplot(1,3,1)
        h = contourf(a_dict.lwc,a_dict.reff,log((loss)'), 15,'LineStyle','none');
        caxis([-6,2])
        colormap 'jet'
        colorbar
        ylabel('Reff [micron]')
        xlabel('lwc [g/kg]')  
        set(gca,'FontName', 'Times New Roman','FontSize',14)
        
        figure (2)
        plot(a_dict.reff(1,:), min(loss,[],2))
        xlabel('Reff [micron]')
        ylabel('min loss per lwc')
        set(gca,'FontName', 'Times New Roman','FontSize',14)
        hold on


%%

for i = 1:size(angles,2)
    figure (1)
    subplot(1,size(angles,2),i)
    subname = num2str(i)+"_SWIR";
    title(subname)
    lege(i) = subname;
end

 savefig("swir_vs_vis[60,30,15,-15,-60].fig")
 
 figure (2)
 legend(lege)
  savefig("swir_vs_vis_loss_min[60,30,15,-15,-60].fig")

        