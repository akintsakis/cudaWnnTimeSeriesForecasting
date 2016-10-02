a=textread('results.txt');
figure

plot(a(:,1));
hold on
plot(a(:,2),'red');