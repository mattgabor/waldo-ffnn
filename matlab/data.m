input_frequencies=[1605;9];
figure(1);
hold on;
for i=1:2
    h=bar(i-1,input_frequencies(i));
    if( i == 1 )
        set(h,'FaceColor','b');
    else
        set(h,'FaceColor','r');
    end
end
set(gca,'Xtick',0:1,'XTickLabel',{'Not Waldo','Waldo'},'FontSize',16);
ylabel('Frequency');
hold off;