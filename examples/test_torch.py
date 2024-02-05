"""
Define custom derivatives via JVPs (forward mode).

The result from this implementation is not too accurate, since
centered finite differences method is implemented.


"""

import numpy as np
import control as ct
import torch
from torch.autograd import Function
import torch.autograd.gradcheck as gradcheck


def model(params, ):
    """
    Create state-space representation and return the system transfer function.
    - params: parameters for the state-space matrices. See https://github.com/oselin/gradient_ML_LTI/blob/main/gradient_propagation.ipynb
    NOTE: The system transfer function is computed via control library, that does not natively support backpropagation of the gradient.
    """
    A = torch.tensor([[params[0], params[1]],
                      [params[2], params[3]]], dtype=torch.double)
    
    B = torch.tensor([params[4], params[5]], dtype=torch.double)

    C = torch.tensor([params[6], params[7]], dtype=torch.double)

    D = torch.tensor([params[8]], dtype=torch.double)

    G = ct.ss2tf(A, B, C, D)

    return G


def forced_response(trn_fcn, u, time):
    """
    Return a torch tensor containing the forced response of the system to input
    - trn_fcn: transfer function on which to compute the forced resonse
    - u: system input in time domain
    - time: time array during which the input is applied
    """
    output = ct.forced_response(trn_fcn, time, u.detach().numpy()).outputs
    output = torch.tensor(output.copy(), requires_grad=True, dtype=torch.double)
    return output


def impulse_response(trn_fcn, time):#
    """
    Return a torch tensor containing the impulse response of the system
    - trn_fcn: transfer function on which to compute the impulse resonse
    - time: time array during which the impulse response has to be computed
    """
    output = ct.impulse_response(trn_fcn, time).outputs
    output = torch.tensor(output.copy(), requires_grad=True, dtype=torch.double)
    return output


def get_magnitude_torch(tensor):
    """
    Compute the magnitude of a torch value
    - tensor: torch tensor
    """
    if torch.equal(tensor, torch.zeros_like(tensor)):
        return 0  # Magnitude of a zero tensor is 0

    magnitude = int(torch.floor(torch.log10(tensor.abs())).item())
    return magnitude


def grad(f, x, h=None):
    """
    Return the gradient, as list of partial derivatives
    computed via (centered) finite differences.
    - f: function, callable. Function f on which to compute the gradient
    - x: parameter with respect to compute the gradient
    - h: step size for finite differences gradient computation

    NOTE: according to the parameter magnitude, a tailored step size h is required.
    This gradient implementation takes into account that

    NOTE: for parameters with magnitude of 1e4, h=1e-2 is demonstrated to be significant
    """

    grads, hs = [], []
    if (h is None): # h is set to auto
        for x_i in x:
            # Get the magnitude of the parameter
            coeff = get_magnitude_torch(x_i) - 6
            hs.append(float(10**coeff))
    else:
        hs = [h for _ in x]


    for i in range(len(x)):
        # NOTE: the copy of x will be with requires_gradient=False
        x_p = x.clone()
        x_m = x.clone()

        x_p[i] += hs[i]
        x_m[i] -= hs[i]

        dfdi = (f(x_p) - f(x_m))/(2*hs[i])
        grads.append(dfdi)

    return grads


class TransferFunction(Function):
    """
    Extend torch.autograd capabilities by designing a custom class TransferFunction that inherits from torch.autograd.Function.
    This allows to manually define both forward and backward methods.
    The gradient is propagated in the backward method via JVP
    """
    @staticmethod
    def forward(ctx, function_input, u, time):

        # Direct computation: compute the forward operation i.e the output of the transfer function
        output = forced_response(model(function_input), u, time)

        # Save the current input and output for further computation of the gradient
        ctx.save_for_backward(function_input, u, time) 
             
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Try to bind the output gradient with the input gradient. i.e. chain rule

        f_input, u, time, = ctx.saved_tensors

        # Create g(x) where x are the params.
        # This allows to test the function by manually changing the single parameter
        # See grad function to understand why
        gx = lambda p: forced_response(model(p), u, time)
                
        # Compute the gradients wrt each parameter
        grads = grad(gx, f_input, h=1e-3)

        # Apply the chain rule for each partial derivative to update each parameter p
        out = [grad_output*i for i in grads]
        
        # Convert the output from a list of partial derivatives to a N-by-p matrix, with p number of parameters, N size of data over time
        out = torch.stack(out, dim=1)

        # sum all the gradients to match the needed output dimension, i.e. p
        out = torch.sum(out, dim=0)

        return out, None, None

 

def test():
    # Definition of time, input, parameters, ground truth
    time = torch.tensor(np.linspace(1, 10, 101, endpoint=False))
    u    = torch.sin(time).requires_grad_(True)

    ref_params = torch.tensor([-1, 1, 3, -4, 1, -1, 0, 1, 0], requires_grad=True, dtype=torch.double)


    # The extended class has to be called via the .apply method.
    # It is easier to assign it to an intermediate variable
    myTransferFunction = TransferFunction.apply   
    
    test_passed =gradcheck(myTransferFunction, (ref_params.requires_grad_(True), u.requires_grad_(False), time.requires_grad_(False)))


if __name__ == "__main__":
    test()