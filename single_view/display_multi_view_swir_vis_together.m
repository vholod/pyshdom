clc
clear

wave = ["[0.555]","[1.6]"];
angles =  ["60.0","30.0","25.0","15.0", "0.0"];
name = "50_50_angles_and_waves_sz180_";
%ch = [0 ,0, 0,0,0, 1];
%%

for i = 1:size(angles,2)
        ch = zeros(size(angles));
        ch(i) = 1;
        figure (1)
        loss = zeros(50,50);
        
        for j = 1:size(angles,2)
           load(name+wave(1)+"_"+angles(j)+".mat");
           loss = loss + a_dict.loss;
           if ch(j)
               load(name+wave(2)+"_"+angles(j)+".mat");
               loss = loss + a_dict.loss;
           end
        end
        subplot(1,size(angles,2),i)
        h = contourf(a_dict.lwc,a_dict.reff,log((loss)'), 15,'LineStyle','none');
        caxis([-6,2])
        colormap 'jet'
        ylabel('Reff [micron]')
        xlabel('lwc [g/kg]')  
        set(gca,'FontName', 'Times New Roman','FontSize',14)
        
        figure (2)
        plot(a_dict.reff(1,:), min(loss,[],2))
        xlabel('Reff [micron]')
        ylabel('min loss per lwc')
        set(gca,'FontName', 'Times New Roman','FontSize',14)
        hold on

end
%%

for i = 1:size(angles,2)
    figure (1)
    subplot(1,size(angles,2),i)
    subname = "SWIR at "+num2str(angles(i));
    title(subname)
    lege(i) = subname;
end
colorbar

 savefig("swir_with_vis_by_order_sz180[60,30,25,15,0].fig")
 %%
 figure (2)
 legend(lege)
  savefig("swir_with_vis_by_order_loss_min_sz180[60,30,25,15,0].fig")

        