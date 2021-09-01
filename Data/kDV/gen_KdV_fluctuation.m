% number of realizations to generate
N = 1;
s = 2048;% grid size

steps = 1;

input = zeros(N, s);

if steps == 1
    output = zeros(N, s);
else
    output = zeros(N, steps, s);
end

tspan = linspace(0,1,steps+1);
x = linspace(0,1,s+1);
lamda = 0.05;

for j=1:N
    u0 = 0.5*randnfun(lamda,[0,1],'trig');
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
end

figure;
plot(input)
title('input')
figure;
plot(output)
title('output')

% save('kdv_fluc_005_2048.mat','input','output');