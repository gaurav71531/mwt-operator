function u = KdV1(init, tspan, s)
        
S = spinop([0 1], tspan);
dt = tspan(2) - tspan(1);
S.lin = @(u) - diff(u,3);
S.nonlin = @(u) -.5*diff(u.^2);
S.init = init;
u = spin(S,s,dt,'plot','off'); 

