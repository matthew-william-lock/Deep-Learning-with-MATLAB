s = (rand(1000,1)-0.5)*20;
% s = 2*randn(100,1);
h = (1+exp(-s)).^-1;
% h_bar = diag(h.*(1-h));
h_bar = diag(h) - diag(diag(h*h'));

figure;
plot(s,h,'.');
hold on;
plot(s,diag(h_bar),'.');

legend('\sigma(s)',"\sigma(s)'")
xlabel('s = Wx + b')
ylabel('Valueb')
