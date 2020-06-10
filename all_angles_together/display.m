clc
clear

wave = ["[0.672]", "[1.6]","both"];
angles = ["[60.0, 60.0, 0.0, 60, 60]","[70.5, 45.6, 0.0, -45.6, -70.5]"];
shape = ["string of pearls", "circle"];
name = "50_50_all_angles";

%%
for j = 2:2        
    fig = figure (j);
    for i = 1:3
        load(name+"_"+wave(i)+"_"+angles(j)+".mat");

        subplot (1,3,i)
        h = contourf(a_dict.lwc,a_dict.reff,log((a_dict.loss)'), 15,'LineStyle','none');
        caxis([-5,2])
        colormap 'jet'
        colorbar
        xlabel('lwc [g/kg]')
        ylabel('Reff [micron]')
        set(gca,'FontName', 'Times New Roman','FontSize',14)
        clear loss
        subplot(1,3,i)
        title(wave(i))
    end

    savefig(shape(j)+".fig")
end


        