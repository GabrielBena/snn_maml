import torch


class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, th=0):
        ctx.save_for_backward(input_)
        return (input_ > th).type(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (10 * torch.abs(input_) + 1.0) ** 2


class ThresholdSurrogate(torch.autograd.Function):
    # need two sigmoids, or maybe 1 - fast sigmoid deriviative
    # similar to fast sigmoid but need  >th and <-th so two directions
    # impulse that goes to zero, look at updated notes up

    # Very similar to FastSigmoid, the difference is the negative threshold
    @staticmethod
    def forward(
        ctx, input_, th
    ):  # not sure what to set the threshold at, will likely need to experiment
        """
         Parameters
        ----------
        input_: The input, which should be dLdS which is the error
        th: the threshold that triggers learning

        Returns
        -------
        thresholded input_
        """
        ctx.save_for_backward(input_, th)

        return input_ * (
            (input_ > th).type(input_.dtype) + (input_ < -th).type(input_.dtype)
        )

    # this is FastSigmoid derivative
    @staticmethod
    def backward(ctx, grad_output):
        (input_, th) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return (input_ > th).type(input_.dtype) + (input_ < -th).type(
            input_.dtype
        )  # (grad_input / (10 * torch.abs(input_) + 1.0) ** 2)
