# PINN - Heat Equation

Physics-Informed Neural Network (PINN) model for simulating the dissipation of a heat source in time and space according to the Fourier equation.


The file `HE_PINN` is organized into the following sections:

- **Training dataset creation**
- **Neural network (NN) architecture definition**
- **NN training using ADAM optimizer**
- **Trining plots**
- **Validation plots**
- **Error analysis**

## Implemented Functions

The file `modelLoss` includes the following main functions:

- `model_loss`: main function for computing the total loss  
- `pinn_loss`: PDE-based loss function  
- `lossIC`: loss for initial condition enforcement
  ```matlab
     alpha = 5^2;
  T0 =exp(-alpha*x.^2);
  ```
- `lossBC`: loss for boundary condition enforcement (Dirichlet conditions)


The function `heat_solution` calculates the numerical solution of the partial differential equation with the command pdepe that needs the following main functions:

- `pde_function`: defines the partial differential equation (PDE).
- `initial_condition`: defines the initial condition
   ```matlab
      alpha = 5^2;
  T0 =exp(-alpha*x.^2);
  ```
- `boundary_condition`: defines the boundary condition
  ```matlab
  %Dirichlet conditions
   pl = Tl; ql = 0;  % Left condition: T(-1,t) = 0
    pr = Tr; qr = 0;  % Right condition: T(1,t) = 0
  ```
 
