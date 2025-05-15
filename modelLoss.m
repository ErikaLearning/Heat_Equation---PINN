%% LOSS FUNCTION
function [loss, grad, lossPinn, lossIC, lossData, lossBC, residual, T, Tt, Tx, Txx]= modelLoss(net, tdata, xdata, inputData,BC_left,BC_right,TBC, inputIC, T0IC, T_analytical) 
[lossPinn, residual,Tt, Tx, Txx]= pinnsLoss(net, inputData);
T = forward(net, inputData);
% xdata = xdata(:);                        % Nx1
% [xgrid, tgrid] = meshgrid(xdata, tdata);        % X, T saranno 10000xN
% inputData_nxn = [xgrid(:)'; tgrid(:)'];          % 2 x (10000*N)
% inputData_nxn = dlarray(inputData_nxn, 'CB');
% 
% T = reshape(T, [length(xdata), length(tdata)]);
% %PINN loss
 
%Loss initial condition
lossIC = ICloss(net, inputIC, T0IC);

T=reshape(T,100,100);
lossData = mean((T - T_analytical).^2, 'all');
lossBC=BCloss(net,BC_left,BC_right,TBC);
%Total loss and gradients
% loss=30*lossPinn+10*lossIC+0.01*lossData+lossBC;
loss=lossPinn+lossIC+lossData+lossBC;
% loss=0.25*lossPinn+0.85*lossIC+lossData+0.1*lossBC;
grad=dlgradient(loss, net.Learnables);

 % Stampiamo i valori
     % disp(['MSE tot: ', num2str(loss)]);
    % disp(['Loss PDE: ', num2str(lossPinn)]);
    % disp(['Residual mean: ', num2str(mean(residual(:)))]);
    % disp(['Tt mean: ', num2str(mean(Tt(:)))]);
    % disp(['Tx mean: ', num2str(mean(Tx(:)))]);
    % disp(['Txx mean: ', num2str(mean(Txx(:)))]);
   %disp(['Loss Analitics: ', num2str(lossAnalytical)]);
   % disp(['Gradient norm: ', num2str(norm(extractdata(grad)))]);
end 


%% PINN loss function
function [lossPinn, residual, Tt, Tx, Txx]= pinnsLoss(net, inputData,T) %+coefficienti
k=0.1;
%T learned from network
T = forward(net, inputData);
%First order derivatives
gradientsU = dlgradient(sum(T, 'all'), inputData, 'EnableHigherDerivatives', true);
    Tx = reshape(gradientsU(1, :), size(T));  % dT/dt
    Tt = reshape(gradientsU(2, :), size(T));  % dT/dx
%Second order derivatives
    grads2 = dlgradient(sum(Tx(:)), inputData, 'EnableHigherDerivatives', true);
    Txx = reshape(grads2(1, :), size(T));   %d²T/dx²

%Residual=PDE=0
residual=Tt-k.*Txx;

%MSE
lossPinn=mean(residual.^2, 'all'); 
end

%%  INITIAL CONDITION LOSS FUNCTION
function lossIC = ICloss(net, inputIC, T0IC)
    % t0 = zeros(size(x0IC));  
    % inputIC = [x0IC; t0];  %2x20

    % initial T learned from network
    TIC_pred = forward(net, inputIC);  
    
    %MSE
    lossIC = mean((TIC_pred - T0IC).^2, 'all');  
end


%% BOUNDARY CONDITION LOSS FUNCTION 
function lossBC=BCloss(net,BC_left,BC_right,TBC)
 %T boundary learned from network
    TBC_left = forward(net,BC_left );  
    TBC_right = forward(net,BC_right );
    err_left = TBC_left - TBC(1, :);
    err_right = TBC_right - TBC(2, :);

    lossBC = mean(err_left.^2 + err_right.^2, 'all');
    % MSE
    % lossBC=mean((TBC_left-TBC(1,:).^2)+((TBC_right-TBC(2,:)).^2), 'all');
end


% %% PDE - Numerical solution
% function T_numerical = heat_solution(xdata,tdata)
% %data preparation
% % tdata = extractdata(tdata); % Converte da dlarray a double
% % xdata = extractdata(xdata); % Converte da dlarray a double
% if isa(xdata, 'dlarray')
%     xdata = double(extractdata(xdata));
% end
% 
% if isa(tdata, 'dlarray')
%     tdata = double(extractdata(tdata));
% end
% 
% xdata = unique(xdata, 'sorted');  % Rimuove duplicati e ordina
% tdata = unique(tdata, 'sorted'); % Ordina tdata ed elimina duplicati
% xdata = xdata(:)'; 
% tdata = tdata(:)';
% 
% %PDE resolution
% T_numerical = pdepe(0, @pde_function, @initial_condition, @boundary_condition, xdata, tdata);
% end
% 
% 
% %PDE Function
%     function [c, f, s] = pde_function(x, t, T, dTdx)
%     k = 0.1; % Thermal diffusion coefficient
%     c = 1; % Multiplicative coefficient of the time derivative
%     f = k.* dTdx; % Diffusive flux
%     s = 0; % Thermal source (absent)
% end
% 
% % PDE - Initial condition
% function T0 = initial_condition(xdata)
%     alpha = 5^2; 
%     T0 = exp(-alpha .* xdata.^2);
% end
% 
% % PDE - Boundary condition
% function [pl, ql, pr, qr] = boundary_condition(xl, Tl, xr, Tr, t)
%     pl = Tl; ql = 0;  % Left condition: T(-1,t) = 0
%     pr = Tr; qr = 0;  % Right condition: T(1,t) = 0
% end






% 
% T = extractdata(T);T_analytical = extractdata(T_analytical);xdata = extractdata(xdata);tdata = extractdata(tdata);
% figure;imagesc(xdata, tdata, T_analytical); %x=0, tutto range di t
% set(gca, 'YDir', 'normal');
% colorbar;
% figure;imagesc(xdata, tdata, T); %x=0, tutto range di t
% set(gca, 'YDir', 'normal');
% colorbar;