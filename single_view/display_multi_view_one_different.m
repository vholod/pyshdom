clc
clear

wave = ["[1.6]","[0.555]"];
angles =  ["60.0","30.0","25.0","15.0","0.0"];%,"-15.0","-60.0"];
name = "50_50_angles_and_waves_sz180_";

%%

for i = 1:size(angles,2)   
        figure (1)
        loss = zeros(50,50);
        
        load(name+wave(1)+"_"+angles(i)+".mat");
        loss = loss + a_dict.loss;
        for k = 1:size(angles,2)
            if (k~=i)
                load(name+wave(2)+"_"+angles(k)+".mat");
                loss = loss + a_dict.loss;
            end
        end
        subplot(1,size(angles,2),i)
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
        clear loss
end

for i = 1:size(angles,2)
    figure (1)
    subplot(1,size(angles,2),i)
    subname = "SWIR in "+angles(i);
    title(subname)
    lege(i) = subname;
end


 savefig("swir_vs_vis_single swir_sz180[60,30,15,0].fig")
 
 figure (2)
 legend(lege)
  savefig("swir_vs_vis_single_swir_loss_min_sz180[60,30,15,0].fig")

        