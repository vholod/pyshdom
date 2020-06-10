clc
clear

wave = ["[0.555]", "[0.672]","[1.6]","both_[0.555]", "both_[0.672]"];
angles = ["[60.0, 60.0, 0.0, 60, 60]","[70.5, 45.6, 0.0, -45.6, -70.5]"];
shape = ["circle", "string of pearls"];
name = "50_50_all_angles";

%%
for j = 2:2        
    for i = 1:5
        fig = figure (j);
        load(name+"_"+wave(i)+"_"+angles(j)+".mat");
        m =  min(a_dict.loss,[],2);
        subplot (1,5,i)
        h = contourf(a_dict.lwc,a_dict.reff,log((a_dict.loss)'), 15,'LineStyle','none');
        caxis([-5,2])
        colormap 'jet'
        colorbar
        xlabel('lwc [g/kg]')
        ylabel('Reff [micron]')
        set(gca,'FontName', 'Times New Roman','FontSize',14)
        clear loss
        subplot(1,5,i)
        title(wave(i))
        figure (3)
        plot(m)
        hold on
        set(gca,'FontName', 'Times New Roman','FontSize',14)
    end

    savefig(shape(j)+".fig")
end


        