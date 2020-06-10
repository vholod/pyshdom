clc
clear

wave = ["[0.672]", "[1.6]","both"];
angles = ["70.5","45.6","0.0","-45.6","-70.5"];
name = "50_50_angles_and_waves_";

%%
fig = figure ();

for i = 1:3       

    for j = 1:5
        load(name+wave(i)+"_"+angles(j)+".mat");
        subplot (3,5,5*(i-1)+j)
        h = contourf(a_dict.lwc,a_dict.reff,log((a_dict.loss)'), 15,'LineStyle','none');
        caxis([-6,2])
        colormap 'jet'
        colorbar
        if (i ==2 && j==1)
           ylabel('Reff [micron]')
        elseif (i == 3 && j==3) 
           xlabel('lwc [g/kg]')  
        end
        set(gca,'FontName', 'Times New Roman','FontSize',14)
        clear loss

    end

   
end

for i = 1:5
    subplot(3,5,i)
    title(angles(i))
end


 savefig("single_viewing_angles.fig")

        