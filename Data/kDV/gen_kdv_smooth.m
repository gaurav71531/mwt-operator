% number of realizations to generate
N = 1;
s = 8192;
% parameters for the Gaussian random field
gamma = 2.5;
tau = 7;
sigma = 7^(2);

steps = 1;

input = zeros(N, s);

if steps == 1
    output = zeros(N, s);
else
    output = zeros(N, steps, s);
end

tspan = linspace(0,1,steps+1);
x = linspace(0,1,s+1);
for j=1:N
    u0 = GRF_kdv(s/2, 0, gamma, tau, sigma);
    u = KdV1(u0, tspan, s);
    
    u0eval = u0(x);
    input(j,:) = u0eval(1:end-1);
    
    if steps == 1
        output(j,:) = u.values;
    else
        for k=2:(steps+1)
            output(j,k,:) = u{k}.values;
        end
    end
    
%     disp(j);
end

figure;
plot(input)
title('input')
figure;
plot(output)
title('output')
% save('kdv_train_test.mat','input','output');
