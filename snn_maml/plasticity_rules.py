import torch

from collections import OrderedDict
from torchmeta.modules import MetaModule
from snn_maml.sigmoid import FastSigmoid, ThresholdSurrogate

import torch.nn.functional as F

import pdb

from matplotlib import pyplot as plt
from torchviz import make_dot
from pathlib import Path
import os
from torch.nn import Parameter

# class FastSigmoid(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_,th=0):
#         ctx.save_for_backward(input_)
#         return  input_ / (1+torch.abs(input_))

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input_,) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         return grad_input / (torch.abs(input_) + 1.0) ** 2#, None

fast_sigmoid = FastSigmoid.apply

error_trigger = ThresholdSurrogate.apply


def grad_flow(path, grad):
    # helps monitor the gradient flow
    # pdb.set_trace()
    grad_norm = [torch.norm(g).item() / torch.numel(g) for g in grad]

    plt.figure()
    plt.semilogy(grad_norm)
    plt.savefig(path + "gradFlow_inner_D.png")
    plt.close()


def cross_entropy_gradient(S, targets):
    # assuming softmax function
    # this is same as Ei
    # print("cross entropy")

    S.retain_grad()
    loss = torch.mean(-torch.sum(targets * torch.log(F.softmax(S, dim=1)), dim=1))
    loss.backward(retain_graph=True)
    return S.grad


def custom_sgd(
    model,
    loss,
    params=None,
    step_size=0.5,
    first_order=False,
    custom_update_fn=None,
    save_graph=False,
):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.

    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.

    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.

    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.

    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError(
            "The model must be an instance of `torchmeta.modules."
            "MetaModule`, got `{0}`".format(type(model))
        )

    if params is None:
        # print("params is None")
        params = OrderedDict(model.meta_named_parameters())
        # print("params", params.keys())
    # else:
    #     print("params is not None")
    #     print(params.keys())

    if save_graph:
        path = Path(f"{os.getcwd()}/graphs/input_{model.i}/", parents=True, exist_ok=False)
        make_dot(loss, params).render(str(path) + "/loss_graph")

    grads = torch.autograd.grad(
        loss,
        params.values(),
        create_graph=not first_order,
        allow_unused=True,
    )
    for n, g in zip(params.keys(), grads):
        if g is None:
            print(f"grad is None for {n} at input {getattr(model, 'i', None)}")
        elif not g.any():
            print(f"grad is zero for {n} at input {getattr(model, 'i', None)}")

    if save_graph and None not in grads:
        grad_flow(str(path) + "/", grads)

    # pdb.set_trace()

    #     torch.save(grads,"saved_inputs_and_grads/grads.pt")

    #     i=1/0

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            if grad is not None:
                if custom_update_fn is not None and "weight" in name:
                    deltaw = custom_update_fn(grad, params[name].data, eta=step_size[name])
                    updated_params[name] = param - deltaw  # ws - w - ws
                else:
                    updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            if grad is not None:
                # print(f"{name} grad is")
                # print(grad.shape)
                # print(grad)
                # pdb.set_trace()
                if custom_update_fn is not None and "weight" in name:
                    w = custom_update_fn(grad, param, eta=step_size)
                    updated_params[name] = w
                else:
                    # if grad.any()!=0:
                    #     print('updating')
                    #     pdb.set_trace()
                    # updated_param = param.clone()
                    updated_param = param - step_size * grad
                    # if isinstance(param, Parameter):
                    #     updated_param = Parameter(updated_param)
                    updated_params[name] = updated_param
            else:
                updated_params[name] = param
                print(f"grad is None for {name} at input {model.i}")

    return updated_params


def custom_sgd_reg(model, loss, params=None, step_size=0.5, anchor_params=None, lamda=1.0):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.
    """
    if not isinstance(model, MetaModule):
        raise ValueError(
            "The model must be an instance of `torchmeta.modules."
            "MetaModule`, got `{0}`".format(type(model))
        )

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    anchor_params = OrderedDict(anchor_params)

    grads = torch.autograd.grad(loss, params.values(), create_graph=True, allow_unused=True)

    updated_params = OrderedDict()

    for (name, param), grad in zip(params.items(), grads):
        if isinstance(step_size, (dict, OrderedDict)):
            ss = step_size[name]
        else:
            ss = step_size

        # Should this not be the difference between param and anchor_value? Note the negative sign inside the bracket
        updated_params[name] = param - ss * lamda * (param - anchor_params[name])
        # updated_params[name] = (1 - lamda * ss) * param - lamda * ss * anchor_params[name]

        if grad is not None:
            updated_params[name] = param - step_size[name] * grad

    return updated_params, grads


def loihi_soel(
    model,
    inputs,
    target,
    params=None,
    step_size=0.01,
    first_order=False,
    learning_engine=None,
):
    """
    Update last layer with SOEL using Loihi Plasticity

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
             The model.

    logits: torch.tensor() float or int
    The model output, can be sum of spikes, I've had most succes with that on non-meta
    but voltage should be able to work to because the plasticity trys to spike like the
    loihi would and I would use those spikes but the spikes are based on the output so
    may need to experiment to see what is best

    target: torch.tensor() int
    Tensor containing the integer value of the target classe.
    This indicates which neuron should be learning when.
    Basically, zero grad non-target neurons

    params : `collections.OrderedDict` instance, optional
    Dictionary containing the meta-parameters of the model. If `None`, then
    the values stored in `model.meta_named_parameters()` are used. This is
    useful for running multiple steps of gradient descent as the inner-loop.

    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.

    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.

    learning_engine: LoihiPlasticity
    Implements the learning rule using a model for loihi's plasticity processor

    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the SOEL Loihi Plasticity learning rule weight updates
    """

    # can only process one sample at a time? I guess so with how traces work
    # maybe if I combine with sgd I can train whole nets with it?

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    # soel_grads = torch.zeros((model.blocks[-1].synapse.weight.shape[0], model.blocks[-1].synapse.weight.shape[1])).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # pdb.set_trace()

    for inp in range(target.shape[0]):
        with torch.no_grad():  # assuming only inner loop needed here (won't work in outer loop)
            logits = model(inputs[inp].unsqueeze(0), params=params)
        thresh = 0
        for i in range(
            5
        ):  # this should simulate tEpoch=20 I think???? only works if time is 100, I've been using 100 but not flexible
            # pdb.set_trace()
            err = 10 - torch.sum(
                learning_engine.y[0][0][target[inp]].T[20 * i : 20 * (i + 1)].T, axis=-1
            )
            # pdb.set_trace()
            if err != 0 and ((err > thresh) or (err < -thresh)):
                learning_engine.y[1][0][target[inp]][20 * (i + 1) - 1] = (
                    20 + err if (20 + err) >= 0 else 0
                )

                thresh += 1
            else:
                if thresh > 0:
                    thresh -= 1

    # pdb.set_trace()
    learning_engine.apply()  # applies the learning to update the weights based on grad values (traces)

    # pdb.set_trace()
    updated_params = OrderedDict()

    names = [name for name, param in params.items()]
    for name, param in params.items():
        # pdb.set_trace()
        if name != names[-1]:
            updated_params[name] = param  # - step_size * grad
        else:
            # pdb.set_trace()
            updated_params[name] = param + step_size * model.blocks[-1].synapse.weight.grad

    # pdb.set_trace()
    print(f"grads {model.blocks[-1].synapse.weight.grad.squeeze()}")
    print(
        f"loihi quantized before: {model.blocks[-1].synapse.pre_hook_fx(params[name],descale=True).squeeze()}"
    )
    print(
        f"loihi quantized after: {model.blocks[-1].synapse.pre_hook_fx(updated_params[name],descale=True).squeeze()}"
    )
    # pdb.set_trace()
    return updated_params


def maml_soel(model, U, targets, params=None, step_size=0.5, first_order=False, threshold=None):
    """Update last layer with SOEL algorithm from Stewart et.al 2020 JETCAS
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.

    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.

    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.

    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.

    threshold : torch.Tensor, either 1 dim or dim compatible with last layer output. (default: torch.Tensor([.05], requires_grad=False)

    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.

    """

    if not isinstance(model, MetaModule):
        raise ValueError(
            "The model must be an instance of `torchmeta.modules."
            "MetaModule`, got `{0}`".format(type(model))
        )

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    pdb.set_trace()

    S = fast_sigmoid(U)

    # dLdU = cross_entropy_gradient(U,targets)

    dLdS = cross_entropy_gradient(S, targets)
    dLdS.requires_grad = True

    # dLdU.requires_grad = True
    # dSdU = torch.autograd.grad(S,U,dLdS,retain_graph=True)#S.backward(U, retain_graph=True)
    if threshold is None:
        threshold = torch.Tensor([0.05], requires_gradient=False).to(model.get_input_layer_device())
    triggered = error_trigger(dLdS, threshold)

    # USE THIS FOR DEBUGGING
    # print('Error Rate',torch.sum(triggered!=0)/torch.prod(torch.Tensor(tuple(triggered.size()))))

    ## Update last layer only (as in SOEL paper )
    if hasattr(model, "LIF_layers"):
        param_name_weight = "LIF_layers.{0}.base_layer.weight".format(len(model.LIF_layers) - 1)
        param_name_bias = None  #'LIF_layers.{0}.base_layer.bias'.format(len(model.LIF_layers)-1)

        dUdW = torch.autograd.grad(
            U, params[param_name_weight], grad_outputs=triggered, retain_graph=True
        )
        if param_name_bias is not None:
            dUdb = torch.autograd.grad(
                U, params[param_name_bias], grad_outputs=triggered, retain_graph=True
            )

    elif hasattr(model, "blocks"):
        mhid = "Mhid"
        param_name_weight = f"blocks.{len(model.network_params[mhid])}.synapse.weight"
        param_name_bias = None

        dUdW = torch.autograd.grad(
            U, params[param_name_weight], grad_outputs=triggered, retain_graph=True
        )
        # dUdb = torch.autograd.grad(U,params[param_name_bias],grad_outputs=triggered,retain_graph=True)

    # pdb.set_trace()

    updated_params = OrderedDict()

    for name, param in params.items():
        if name == param_name_weight:
            updated_params[name] = param - step_size * dUdW[0]
        elif name == param_name_bias:
            updated_params[param_name_bias] = param - step_size * dUdb[0]
        else:
            updated_params[name] = param

    return updated_params
