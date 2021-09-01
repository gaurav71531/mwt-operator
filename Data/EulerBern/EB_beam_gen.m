num = 1;
s = 1024;
w = 215;
lamda = 0.2;
F = 5e3;
N = chebop([0,1]);
N.op = @(x,u) diff(u,4)-w^2.*u;

N.lbc = @(u) [u;diff(u)];
N.rbc = @(u) [u;diff(u)];
x = linspace(0,1,s+1);
input = zeros(num, s);
output = zeros(num,s);
%% Force function
for i = 1:num
    f = randnfun(lamda,[0,1],'trig');  %randn
    f = (f + 2)*F;

    u = N\f;
    u_output = u(x);
    output(i,:) = u_output(1:end-1);
    a = f/F; % input
    a_input = a(x);
    input(i,:) = a_input(1:end-1);
figure;
subplot(1, 2, 1);plot(input);
subplot(1, 2, 2);plot(output);
end
% save('EB_beam_1024_02_215.mat','input','output');
