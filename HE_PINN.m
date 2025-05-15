clc; clear all; close all;
%% dataset
numInitialConditionPoints  = 40;
numBoundaryConditionPoints = [30 30]; %estremi dello spazio (x=−1 e x=1),
numInternalCollocationPoints = 1e4;

%spatial and temporal points
x_p=[-1:0.0002:1];
t_p= (x_p+1)./2;
points=[t_p' x_p'];
points=points(1:10000,:);

xmin = -1;
xmax =  1;
tmax =  1;
% Set the value of x for the Boundary Conditions (Dirichlet)
%T=0 for x=1 and x=-1 at every t
x0BC1 = xmin*ones(1,numBoundaryConditionPoints(1)); %vettore riga -1
x0BC2 = xmax*ones(1,numBoundaryConditionPoints(2)); %vettore riga +1
t0BC1 = linspace(0,tmax,numBoundaryConditionPoints(1)); 
t0BC2 = linspace(0,tmax,numBoundaryConditionPoints(2));
T0BC1 = zeros(1,numBoundaryConditionPoints(1));
T0BC2 = zeros(1,numBoundaryConditionPoints(2));

% Set the value for the Initial Conditions
t0IC = zeros(1,numInitialConditionPoints);
x0IC = linspace(xmin,xmax,numInitialConditionPoints);
%Source of heat at x=0, as a gaussian/bell shape
alpha = 5^2;
T0IC =exp(-alpha*x0IC.^2); 

% Boundary condition in vector form
BC_left=[x0BC1;t0BC1]';
BC_right=[x0BC2;t0BC2]';
TBC = [T0BC1;T0BC2];

% Initial condition in vector form
t0 = zeros(size(x0IC));  
inputIC = [x0IC; t0];  %2x20

%TRAINING DATASET
tdata=points(:,1);
xdata=points(:,2);
inputData = [xdata tdata]';
% T_analytical = heat_solution(xdata, tdata);

% t_entr = linspace(0, 1, 100);           % 1x100
x_entr=[-1:0.02:1];                       
x_entr=x_entr(:,1:100); % 1x100
t_entr= (x_entr+1)./2;
% points=[t_p' x_p'];
% points=points(1:10000,:);
[xgrid, tgrid] = meshgrid(x_entr, t_entr);        % X, T saranno 10000xN
inputData_entr = [xgrid(:)'; tgrid(:)'];          % 2 x (10000*N)
inputData = dlarray(inputData_entr, 'CB');
T_analytical = heat_solution(x_entr, t_entr);

%% Neural Network
inputsize=2; %x,t
outputsize=1; %T
hiddenunits=10;
numLayers = 10; 

% Input layer
layers = [featureInputLayer(inputsize, 'Name', 'input')];
% Hidden layers
for i = 2:numLayers-1
    layers = [layers 
              fullyConnectedLayer(hiddenunits, 'Name', ['fc' num2str(i)]) 
               tanhLayer('Name', ['tanh' num2str(i)])];
                % reluLayer('Name', ['relu' num2str(i)])];
end
% Output layer
layers = [layers 
          fullyConnectedLayer(outputsize, 'Name', 'output')]; 

%Deep learning net
net = dlnetwork(layers);
% deepNetworkDesigner(net)

%% Train the NN
%with ADAM optimizer
%ADAM hyperparameter
learnrate=0.01;
mp=[]; %mean term
vp=[]; %variance term
numinterations=150;

%Training dataset in dlarray
xdata = dlarray(xdata','CB');
tdata = dlarray(tdata','CB');
TBC = dlarray(TBC,'CB');
x0IC = dlarray(x0IC,'CB');
T0IC = dlarray(T0IC,'CB');
% inputData = dlarray(inputData, 'CB');
BC_right=dlarray(BC_right', 'CB');
BC_left=dlarray(BC_left', 'CB');
inputIC=dlarray(inputIC, 'CB');


%training progress plot
monitor= trainingProgressMonitor(Metrics=["Loss","LossPDE", "LossBC", "LossIC","LossData"]);
%fig=figure(Visible="off");
% fig=figure();

%Acceleration del model loss
accFnc= dlaccelerate(@modelLoss); %LOSS FUNCTION
accFnc2=dlaccelerate(@BCloss); %BC LOSS FUNCTION

for iteration=1:numinterations
    %differenziazione automatica
    % [loss, grad, lossPinn, lossAnalytical, lossBC, lossIC, residual, Tt, Tx, Txx]=dlfeval(accFnc, net, tdata, xdata, inputData,BC_left,BC_right,TBC, x0IC, T0IC); %, x0IC, t0IC,T0IC,XBC, TBC,tBC); %+coefficienti vari
    %, x0IC, t0IC,T0IC,XBC, TBC,tBC);

    
    [loss, grad, lossPinn, lossIC, lossData,lossBC, residual,T, Tt, Tx, Txx]=dlfeval(accFnc, net, t_entr, x_entr, inputData,BC_left,BC_right,TBC, inputIC, T0IC, T_analytical); %, x0IC, t0IC,T0IC,XBC, TBC,tBC);

    
    %update the model
    [net,mp,vp]=adamupdate(net,grad,mp,vp, iteration, learnrate);
    
    % training progress metrics names
  recordMetrics(monitor, iteration, ...
        Loss=loss, ...
        LossPDE=lossPinn, ...
        LossBC=lossBC, ...
        LossIC=lossIC, ...
        LossData=lossData)
end
disp(['MSE tot in training: ', num2str(loss)]);
disp(['MSE BC in training: ', num2str(lossBC)]);
disp(['MSE IC in training: ', num2str(lossIC)]);
disp(['MSE PINN in training: ', num2str(lossPinn)]);
disp(['MSE DATA in training: ', num2str(lossData)]);

%% GRAPHS TRAINING


% Prepara t e x come griglie da combinare
% t_entr = linspace(0, 1, 100);           % 1x10000
% x_entr=[-1:0.02:1];                       % Nx1
% x_entr=x_entr(:,1:100);
% 
% [xgrid, tgrid] = meshgrid(x_entr, t_entr);        % X, T saranno 10000xN
% 
% inputData_entr = [xgrid(:)'; tgrid(:)'];          % 2 x (10000*N)
% inputData_entr = dlarray(inputData_entr, 'CB');
% Predizione in batch
% T_pred_entr = predict(net, inputData);
T_pred_entr = predict(net, inputData);    % output: 1 x (10000*N)
T_pred_entr = extractdata(T_pred_entr);             % double

% Ricostruzione in matrice 10000 x N
T_pred_entr = reshape(T_pred_entr, [length(t_entr), length(x_entr)]);


% Calcoliamo la soluzione analitica
T_true_entr = heat_solution(x_entr,t_entr);
 %T_true_entr = funHE_analitica(xdata,tdata);
% xtest = linspace(-1,1,101);
% ttest=[0:0.01:1];
if isa(tdata, 'dlarray')
    tdata = extractdata(tdata);
end

if isa(xdata, 'dlarray')
    xdata = extractdata(xdata);
end

if isa(T_pred_entr, 'dlarray')
    T_pred_entr = extractdata(T_pred_entr);
end

figure;
sgtitle('Training');


subplot(1,2,1);
% xtest=extractdata(xtest);
% t = 0:0.01:1;
imagesc(xdata, tdata, T_pred_entr);
set(gca, 'YDir', 'normal'); % Per allineare l'asse t
colorbar;
title('Predizione PINN');
xlabel('x'); ylabel('t');
xlim([xmin xmax]); ylim([0 tmax]);

% Grafico della soluzione analitica
subplot(1,2,2);
imagesc(xdata, tdata, T_true_entr); %x=0, tutto range di t
% imagesc(xdata, tdata, T_analytical); %
set(gca, 'YDir', 'normal');
colorbar;
title('Soluzione analitica');
xlabel('x'); ylabel('t');
xlim([xmin xmax]); ylim([0 tmax]);


%% GRAPHS VALIDATION

%Validation dataset
%256/2=128, perche' 256 serve per ottenere un quadrato perfetto con il
%reshape
xtest_neg=sort(rand(50,1)).*xmin;
xtest_pos=sort(rand(50,1)).*xmax;
xtest = sort([xtest_neg; xtest_pos]);  % Unisce e poi ordina tutto
% 


% Numerical solution of the PDE
t = linspace(0,1,100);
t=dlarray(t,'CB');
% xtest=xtest(:)';
xtest = dlarray(xtest,'CB');
T_true_val = heat_solution(xtest,t);

[xgrid, tgrid] = meshgrid(xtest, t);        % X, T saranno 10000xN
% 
inputData_val = [xgrid(:)'; tgrid(:)'];          % 2 x (10000*N)
% inputData_val=[xtest; t];
inputData_val = dlarray(inputData_val, 'CB');

% Predizione in batch
T_pred_val = predict(net, inputData_val);    % output: 1 x (10000*N)
T_pred_val = extractdata(T_pred_val);             % double

% Ricostruzione in matrice 10000 x N
T_pred_val = reshape(T_pred_val, 100,100); %sqrt(256)=16
% Graphs PINN predictions
figure;
sgtitle('Validation');
t = extractdata(t); 
xtest = extractdata(xtest);

%Graph PINN solution
subplot(1,2,1);
imagesc(xtest, t, T_pred_val);
set(gca, 'YDir', 'normal'); 
colorbar;
title(' PINN Prediction');
xlabel('x'); ylabel('t');
xlim([xmin xmax]); ylim([0 tmax]);

% Graphs numerical solution
subplot(1,2,2);
imagesc(xtest, t, T_true_val); 
set(gca, 'YDir', 'normal');
colorbar;
title('Numerical solution');
xlabel('x'); ylabel('t');
xlim([xmin xmax]); ylim([0 tmax]);

%calcolo MSE per Validation
MSE_validation1 = mean((T_pred_val - T_true_val).^2, 'all');


err_IC=T_pred_val(:,1)'-T_true_val(1,:);
MSE_IC_validation=mean(err_IC.^2, 'all');

err_left = T_pred_val(1,:) - T_true_val( :,1);
err_right = T_pred_val(100,:) - T_true_val(:,100);
MSE_BC_validation=mean(err_left.^2 + err_right.^2, 'all');
 


% xtest=xtest(:)'; %1xN
% xtest=dlarray(xtest, 'CB');
% t=dlarray(t, 'CB');
[loss, grad, lossPinn, lossIC, lossData,lossBC, residual, Tt, Tx, Txx]=dlfeval(accFnc, net, t, xtest, inputData_val,BC_left,BC_right,TBC, inputIC, T0IC, T_true_val); %, x0IC, t0IC,T0IC,XBC, TBC,tBC);





% disp(['MSE tot in Validation formula: ', num2str(MSE_validation1)]);
disp(['MSE tot in validation con funct: ', num2str(loss)]);

% disp(['MSE BC in Validation formula: ', num2str(MSE_BC_validation)]);
disp(['MSE BC in validation con funct: ', num2str(lossBC)]);

% disp(['MSE IC in Validation formula: ', num2str(MSE_IC_validation)]);
disp(['MSE IC in validation con funct: ', num2str(lossIC)]);
disp(['MSE PINN in validation con funct: ', num2str(lossPinn)]);
disp(['MSE DATA: ', num2str(lossData)]);

%% GRAPHS ERROR
% tTest = dlarray([0.00 0.10 0.25 0.50 0.75 1.00], 'CB');
tTest=[t(1) t(25) t(50) t(75) t(100)];

% Positions of t with the same value as tTest
pos = zeros(size(tTest));  
    for j = 1:length(tTest)
    idx = find(t == tTest(j));
    if ~isempty(idx)
        pos(j) = idx;  
    else
        pos(j) = -1;   % Se no encuentra mismo valore, guarda -1
    end
    end

%pos=[1,50,100,150,200]

figure
T_all=T_true_entr; %matrix 200x200

for i=1:numel(tTest)
    subplot(3,2,i)
    hold on
    %Predicted values
    T_pred=[];
    ttest = tTest(i).*ones(length(xtest),1);
    inputData_test = [xtest ttest]';
    inputData_test =dlarray(inputData_test, 'CB');
    T = predict(net, inputData_test);
    T_pred = [T_pred, extractdata(T)'];
    %True values
    T_true_entr=T_all(pos(i),:);


    % Calculate error.
    err = norm(T_pred' - T_true_entr) / norm(T_true_entr);
    % Plot predicted values.
    plot(xtest,T_pred,'-','LineWidth',2);
    ylim([-1.1, 1.1]);
    % Plot true values.
    plot(xtest, T_true_entr, '--','LineWidth',2)
    grid minor
    hold off
    title("t = " + tTest(i) + ", Error = " + gather(err),'Interpreter','latex');
end 
subplot(3,2,1)
legend('PINN Solution','numerical Solution',...
    'Location','best','Interpreter','latex')



