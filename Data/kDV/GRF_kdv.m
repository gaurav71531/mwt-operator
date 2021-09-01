function u = GRF_kdv(N, m, gamma, tau, sigma)

my_const = 2*pi;

my_eigs = sqrt(2)*(abs(sigma).*((my_const.*(1:N)').^2 + tau^2).^(-gamma/2));

xi_alpha = randn(N,1);
alpha = my_eigs.*xi_alpha;

xi_beta = randn(N,1);
beta = my_eigs.*xi_beta;
    
a = alpha/2;
b = -beta/2;
c = [flipud(a) - flipud(b).*1i;m + 0*1i;a + b.*1i];

uu = chebfun(c, [0 1],'trig','coeffs');
u = chebfun(@(t) uu(t - 0.5), [0 1],'trig');

end