clc
clear

%wave = ["[0.672]", "[1.6]","both"];
wave = ["[1.59]","[1.6]","[1.61]","[1.62]"];
angles = "0.0";% ["70.5","45.6","0.0","-45.6","-70.5"];
name = "50_50_angles_and_waves_";

%%

for i = 1:size(wave,2)   
        figure (1)
        load(name+wave(i)+"_"+angles+".mat");
        subplot(1,size(wave,2),i)
        h = contourf(a_dict.lwc,a_dict.reff,log((a_dict.loss)'), 15,'LineStyle','none');
        caxis([-6,2])
        colormap 'jet'
        colorbar
        ylabel('Reff [micron]')
        xlabel('lwc [g/kg]')  
        set(gca,'FontName', 'Times New Roman','FontSize',14)
        
        figure (2)
        plot(min(a_dict.loss,[],2))
        hold on
        clear loss
end

for i = 1:size(wave,2)
    figure (1)
    subplot(1,size(wave,2),i)
    title(wave(i))
end


 savefig("swir_sensitivity.fig")

        