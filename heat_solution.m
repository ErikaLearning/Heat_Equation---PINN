%% PDE - Numerical solution
function T_numerical = heat_solution(xdata,tdata)
%data preparation
% tdata = extractdata(tdata); % Converte da dlarray a double
% xdata = extractdata(xdata); % Converte da dlarray a double
if isa(xdata, 'dlarray')
    xdata = double(extractdata(xdata));
end

if isa(tdata, 'dlarray')
    tdata = double(extractdata(tdata));
end

xdata = unique(xdata, 'sorted');  % Rimuove duplicati e ordina
tdata = unique(tdata, 'sorted'); % Ordina tdata ed elimina duplicati
xdata = xdata(:)'; 
tdata = tdata(:)';

%PDE resolution
T_numerical = pdepe(0, @pde_function, @initial_condition, @boundary_condition, xdata, tdata);
end


%PDE Function
    function [c, f, s] = pde_function(x, t, T, dTdx)
    k = 0.1; % Thermal diffusion coefficient
    c = 1; % Multiplicative coefficient of the time derivative
    f = k.* dTdx; % Diffusive flux
    s = 0; % Thermal source (absent)
end

% PDE - Initial condition
function T0 = initial_condition(xdata)
    alpha = 5^2; 
    T0 = exp(-alpha .* xdata.^2);
end

% PDE - Boundary condition
function [pl, ql, pr, qr] = boundary_condition(xl, Tl, xr, Tr, t)
    pl = Tl; ql = 0;  % Left condition: T(-1,t) = 0
    pr = Tr; qr = 0;  % Right condition: T(1,t) = 0
end
