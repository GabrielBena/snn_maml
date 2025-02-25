import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from snn_maml.utils import quantize_parameters
from collections import OrderedDict
from . import plasticity_rules
from .utils import tensors_to_device, compute_accuracy

__all__ = ["ModelAgnosticMetaLearning", "MAML", "FOMAML"]

# from tensorboardX import SummaryWriter

import pdb

# default `log_dir` is "runs" - we'll be more specific here

get_postfix = lambda pbar: (
    dict([s.split("=") for s in pbar.postfix.split(", ")]) if pbar.postfix is not None else {}
)


def batch_one_hot(targets, num_classes=10):
    one_hot = torch.zeros((targets.shape[0], num_classes))
    # print("targets shape", targets.shape)
    for i in range(targets.shape[0]):
        one_hot[i][targets[i]] = 1

    return one_hot


def undo_onehot(targets):
    not_hot = torch.zeros((targets.shape[0]))

    for i in range(targets.shape[0]):
        not_hot[i] = torch.nonzero(targets[0])[0][0].item()

    return not_hot.to(targets.device)


class ModelAgnosticMetaLearning(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.

    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).

    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.

    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].

    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.

    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.

    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].

    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.

    device : `torch.device` instance, optional
        The device on which the model is defined.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)

    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """

    def __init__(
        self,
        model,
        optimizer=None,
        step_size=0.1,
        first_order=False,
        learn_step_size=False,
        per_param_step_size=False,
        num_adaptation_steps=1,
        num_adaptation_samples=None,
        scheduler=None,
        loss_function=F.cross_entropy,
        custom_outer_update_fn=None,
        custom_inner_update_fn=None,
        device=None,
        boil=False,
        outer_loop_quantizer=None,
        inner_loop_quantizer=None,
    ):
        self.model = model.to(device=device)
        self.outer_loop_quantizer = outer_loop_quantizer
        self.inner_loop_quantizer = inner_loop_quantizer
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.num_adaptation_samples = num_adaptation_samples
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.custom_inner_update_fn = custom_inner_update_fn
        self.custom_outer_update_fn = custom_outer_update_fn

        if per_param_step_size or boil:
            self.step_size = OrderedDict(
                (
                    name,
                    torch.tensor(
                        step_size,
                        dtype=param.dtype,
                        device=self.device,
                        requires_grad=learn_step_size,
                    ),
                )
                for (name, param) in model.meta_named_parameters()
            )
            if boil:
                assert learn_step_size is False, "boil is not compatible with learning step sizes"
                last_layer_names = [k for k in self.step_size.keys()][
                    -2:
                ]  # assumed bias and weight in last layer
                for k in last_layer_names:
                    self.step_size[k] = torch.tensor(
                        0.0,
                        dtype=self.step_size[k].dtype,
                        device=self.device,
                        requires_grad=False,
                    )
                print("step_size", self.step_size)
        else:
            self.step_size = torch.tensor(
                step_size,
                dtype=torch.float32,
                device=self.device,
                requires_grad=learn_step_size,
            )

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group(
                {"params": (self.step_size.values() if per_param_step_size else [self.step_size])}
            )
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault("initial_lr", group["lr"])
                # self.scheduler.base_lrs([group['initial_lr'] for group in self.optimizer.param_groups])

    def get_outer_loss(self, batch, **kwargs):
        if "test" not in batch:
            raise RuntimeError("The batch does not contain any test dataset.")

        stream_mode = kwargs.get("stream_mode", True)
        save_graph = kwargs.get("save_graph", False)

        _, test_targets = batch["test"]
        num_tasks = test_targets.size(0)
        is_classification_task = not test_targets.dtype.is_floating_point
        results = {
            "num_tasks": num_tasks,
            "inner_losses": np.zeros((self.num_adaptation_steps, num_tasks), dtype=np.float32),
            "outer_losses": np.zeros((num_tasks,), dtype=np.float32),
            "mean_outer_loss": 0.0,
        }
        if is_classification_task:
            results.update(
                {
                    "accuracies_before": np.zeros((num_tasks,), dtype=np.float32),
                    "accuracies_after": np.zeros((num_tasks,), dtype=np.float32),
                }
            )

        pbar = kwargs.get("pbar", None)

        mean_outer_loss = torch.tensor(0.0, device=self.device)
        # One task per batch_size
        for task_id, (
            train_inputs,
            train_targets,
            test_inputs,
            test_targets,
        ) in enumerate(zip(*batch["train"], *batch["test"])):

            # print("INPUT SHAPE", train_inputs.shape)
            # print("TARGET SHAPE", train_targets.shape)

            if pbar is not None:
                desc = get_postfix(pbar)
                desc.update({"task": f"{task_id}/{num_tasks}"})
                pbar.set_postfix(desc)

            # Test Before Adaptation
            test_inputs, test_targets = test_inputs.to(self.device), test_targets.to(self.device)
            test_logits = self.model(test_inputs.squeeze().transpose(0, 1), params=None)
            outer_loss = self.loss_function(test_logits, test_targets)
            if isinstance(outer_loss, tuple):
                outer_loss = outer_loss[0]

            if is_classification_task:
                results["accuracies_before"][task_id] = compute_accuracy(
                    test_logits,
                    test_targets,
                    first_spike_fn=getattr(self.model, "first_spike_fn", None),
                )

            # Adaptation
            params, adaptation_results = self.adapt(
                train_inputs,
                train_targets,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                num_adaptation_samples=self.num_adaptation_samples,
                step_size=self.step_size,
                first_order=self.first_order,
                stream_mode=stream_mode,
                save_graph=save_graph,
                pbar=pbar,
            )

            results["inner_losses"][:, task_id] = adaptation_results["inner_losses"]

            # Test After Adaptation and Compute Outer Loss
            with torch.set_grad_enabled(self.model.training):

                test_inputs, test_targets = test_inputs.to(self.device), test_targets.to(self.device)
                test_logits = self.model(test_inputs.squeeze().transpose(0, 1), params=params)
                outer_loss = self.loss_function(test_logits, test_targets)
                if isinstance(outer_loss, tuple):
                    outer_loss = outer_loss[0]

                results["outer_losses"][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results["accuracies_after"][task_id] = compute_accuracy(
                    test_logits,
                    test_targets,
                    first_spike_fn=getattr(self.model, "first_spike_fn", None),
                )
            if pbar is not None:
                desc = get_postfix(pbar)
                desc.update(
                    {
                        "Test Acc": f"{results['accuracies_before'][task_id]} -> {results['accuracies_after'][task_id]}"
                    }
                )
                pbar.set_postfix(desc)

        mean_outer_loss.div_(num_tasks)
        results["mean_outer_loss"] = mean_outer_loss.item()

        return mean_outer_loss, results

    # Inner loop
    def adapt(
        self,
        inputs,
        targets,
        is_classification_task=None,
        num_adaptation_steps=1,
        num_adaptation_samples=None,
        step_size=0.1,
        first_order=False,
        stream_mode=True,
        save_graph=False,
        pbar=None,
    ):
        if is_classification_task is None:
            is_classification_task = not targets.dtype.is_floating_point

        params = OrderedDict(self.model.meta_named_parameters())
        if self.outer_loop_quantizer is not None:
            params = quantize_parameters(params, self.outer_loop_quantizer)

        results = {"inner_losses": np.zeros((num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):

            def process_inputs(inputs, targets, params):
                single_results = {}
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs, params=params)
                if len(targets.shape) == 0:
                    targets = torch.tensor([targets]).to(self.device)

                inner_loss = self.loss_function(logits, targets)
                if isinstance(inner_loss, tuple):
                    inner_loss = inner_loss[0]
                # pdb.set_trace()
                if (step == num_adaptation_steps - 1) and is_classification_task:
                    inner_acc = compute_accuracy(logits, targets)

                # print("updating params...")
                self.model.zero_grad()
                params = plasticity_rules.custom_sgd(
                    self.model,
                    inner_loss,
                    step_size=step_size,
                    params=params,
                    first_order=(not self.model.training) or first_order,
                    custom_update_fn=self.custom_inner_update_fn,
                    save_graph=save_graph,
                )

                if self.inner_loop_quantizer is not None:
                    params = quantize_parameters(params, self.inner_loop_quantizer)

                return (
                    params,
                    inner_loss,
                    inner_acc if is_classification_task else None,
                )

            n_samples = num_adaptation_samples or inputs.size(0)
            indices = np.random.choice(inputs.size(0), n_samples, replace=False)

            if stream_mode:
                results["inner_losses"] = [[] for _ in range(num_adaptation_steps)]
                results["inner_accuracies"] = [[] for _ in range(num_adaptation_steps)]
                for i, (input, target) in enumerate(zip(inputs[indices], targets[indices])):
                    self.model.i = i
                    params, inner_loss, inner_acc = process_inputs(input, target, params)
                    results["inner_losses"][step].append(inner_loss.item())
                    results["inner_accuracies"][step].append(inner_acc)
                results["inner_losses"][step] = np.mean(results["inner_losses"][step])
                results["inner_accuracies"][step] = np.mean(results["inner_accuracies"][step])
            else:
                inputs = inputs.squeeze().transpose(0, 1)
                params, inner_loss, inner_acc = process_inputs(inputs[indices], targets[indices], params)
                results["inner_losses"] = inner_loss.item()
                results["inner_accuracies"] = inner_acc

            if pbar:
                desc = get_postfix(pbar)
                desc.update({"inner_loss": inner_loss.item()})
                desc.update({"inner_acc": inner_acc})

        return params, results

    def train(self, dataloader, max_batches=500, verbose=True, epoch=-1, **kwargs):
        mean_outer_loss, mean_accuracy_af, count, mean_accuracy_bf = 0.0, 0.0, 0, 0.0
        with tqdm(total=max_batches, disable=False, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches, epoch=epoch, pbar=pbar):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results["mean_outer_loss"] - mean_outer_loss) / count
                postfix = {"loss": "{0:.4f}".format(mean_outer_loss)}
                if "accuracies_after" in results:
                    mean_accuracy_af += (np.mean(results["accuracies_after"]) - mean_accuracy_af) / count
                    postfix["after in-loop"] = "{0:.4f}".format(np.mean(mean_accuracy_af))
                if "accuracies_before" in results:
                    mean_accuracy_bf += (
                        np.mean(results["accuracies_before"]) - mean_accuracy_bf
                    ) / count
                    postfix["before in-loop"] = "{0:.4f}".format(np.mean(results["accuracies_before"]))
                # pbar.set_postfix(**postfix)

        mean_results = {"mean_outer_loss": mean_outer_loss}
        if "accuracies_after" in results:
            mean_results["accuracies_after"] = mean_accuracy_af
        if "accuracies_before" in results:
            mean_results["accuracies_before"] = mean_accuracy_bf
        return mean_results

    # Outer loop
    def train_iter(self, dataloader, max_batches=500, epoch=-1, pbar=None, **kwargs):
        if self.optimizer is None:
            raise RuntimeError(
                "Trying to call `train_iter`, while the "
                "optimizer is `None`. In order to train `{0}`, you must "
                "specify a Pytorch optimizer as the argument of `{0}` "
                "(eg. `{0}(model, optimizer=torch.optim.SGD(model."
                "parameters(), lr=0.01), ...).".format(__class__.__name__)
            )
        num_batches = 0
        self.model.train()

        # print(self.model)

        for batch, _ in zip(dataloader, range(max_batches)):

            self.optimizer.zero_grad()

            batch = tensors_to_device(batch, device=self.device)
            outer_loss, results = self.get_outer_loss(batch, pbar=pbar)
            yield results
            # pdb.set_trace()
            outer_loss.backward()
            # pdb.set_trace()
            # self.model.grad_flow('./')
            # pdb.set_trace()
            if self.custom_outer_update_fn is not None:
                self.custom_outer_update_fn(self.model)

            self.optimizer.step()
            if hasattr(self.step_size, "__len__"):
                if len(self.step_size.shape) > 0:
                    for name, value in self.step_size.items():
                        if value.data < 0:
                            value.data.zero_()
                            print("Negative step values detected")

            # if self.custom_outer_update_fn is not None:
            #    from .custom_funs import inplace_clamp_model_weights_asymm
            #    inplace_clamp_model_weights_asymm(self.model)

            if self.scheduler is not None:
                self.scheduler.step()

            num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy_af, count, mean_accuracy_bf = 0.0, 0.0, 0, 0.0
        with tqdm(total=max_batches, disable=False, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches, pbar=pbar):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results["mean_outer_loss"] - mean_outer_loss) / count
                postfix = {"loss": "{0:.4f}".format(mean_outer_loss)}
                if "accuracies_after" in results:
                    mean_accuracy_af += (np.mean(results["accuracies_after"]) - mean_accuracy_af) / count
                    postfix["after in-loop"] = "{0:.4f}".format(np.mean(mean_accuracy_af))
                if "accuracies_before" in results:
                    mean_accuracy_bf += (
                        np.mean(results["accuracies_before"]) - mean_accuracy_bf
                    ) / count
                    postfix["before in-loop"] = "{0:.4f}".format(np.mean(results["accuracies_before"]))
                pbar.set_postfix(**postfix)

        mean_results = {"mean_outer_loss": mean_outer_loss}
        if "accuracies_after" in results:
            mean_results["accuracies_after"] = mean_accuracy_af
        if "accuracies_before" in results:
            mean_results["accuracies_before"] = mean_accuracy_bf

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500, pbar=None, **kwargs):
        num_batches = 0
        self.model.eval()
        for batch, _ in zip(dataloader, range(max_batches)):

            batch = tensors_to_device(batch, device=self.device)
            _, results = self.get_outer_loss(batch, pbar=pbar)
            yield results


MAML = ModelAgnosticMetaLearning


class FOMAML(ModelAgnosticMetaLearning):
    def __init__(
        self,
        model,
        optimizer=None,
        step_size=0.1,
        learn_step_size=False,
        per_param_step_size=False,
        num_adaptation_steps=1,
        scheduler=None,
        loss_function=F.cross_entropy,
        device=None,
    ):
        super(FOMAML, self).__init__(
            model,
            optimizer=optimizer,
            first_order=True,
            step_size=step_size,
            learn_step_size=learn_step_size,
            per_param_step_size=per_param_step_size,
            num_adaptation_steps=num_adaptation_steps,
            scheduler=scheduler,
            loss_function=loss_function,
            device=device,
        )


class Reptile:

    def __init__(self, model, log, params):

        # Intialize Reptile Parameters
        self.inner_step_size = params[0]
        self.inner_batch_size = params[1]
        self.outer_step_size = params[2]
        self.outer_iterations = params[3]
        self.meta_batch_size = params[4]
        self.eval_iterations = params[5]
        self.eval_batch_size = params[6]

        # Initialize Torch Model and Tensorboard
        self.model = model.to(device)
        self.log = log

    def reset(self):

        # Reset Training Gradients
        self.model.zero_grad()
        self.current_loss = 0
        self.current_batch = 0

    def train(self, task):

        # Train from Scratch
        self.reset()

        # Outer Training Loop
        for outer_iteration in tqdm.tqdm(range(self.outer_iterations)):

            # Track Current Weights
            current_weights = deepcopy(self.model.state_dict())

            # Sample a new Subtask
            samples, task_theta = sample(task)

            # Inner Training Loop
            for inner_iteration in range(self.inner_batch_size):

                # Process Meta Learning Batches
                for batch in range(0, len(sample_space), self.meta_batch_size):

                    # Get Permuted Batch from Sample
                    perm = np.random.permutation(len(sample_space))
                    idx = perm[batch : batch + self.meta_batch_size][:, None]

                    # Calculate Batch Loss
                    batch_loss = self.loss(sample_space[idx], samples[idx])
                    batch_loss.backward()

                    # Update Model Parameters
                    for theta in self.model.parameters():

                        # Get Parameter Gradient
                        grad = theta.grad.data

                        # Update Model Parameter
                        theta.data -= self.inner_step_size * grad

                    # Update Model Loss from Torch Model Tensor
                    loss_tensor = batch_loss.cpu()
                    self.current_loss += loss_tensor.data.numpy()
                    self.current_batch += 1

            # Linear Cooling Schedule
            alpha = self.outer_step_size * (1 - outer_iteration / self.outer_iterations)

            # Get Current Candidate Weights
            candidate_weights = self.model.state_dict()

            # Transfer Candidate Weights to Model State Checkpoint
            state_dict = {
                candidate: (
                    current_weights[candidate]
                    + alpha * (candidate_weights[candidate] - current_weights[candidate])
                )
                for candidate in candidate_weights
            }
            self.model.load_state_dict(state_dict)

            # Log new Training Loss
            self.log.add_scalars(
                "Model Estimate/Loss", {"Loss": self.current_loss / self.current_batch}, outer_iteration
            )

    def loss(self, x, y):

        # Reset Torch Gradient
        self.model.zero_grad()

        # Calculate Torch Tensors
        x = torch.tensor(x, device=device, dtype=torch.float32)
        y = torch.tensor(y, device=device, dtype=torch.float32)

        # Estimate over Sample
        yhat = self.model(x)

        # Regression Loss over Estimate
        loss = nn.MSELoss()
        output = loss(yhat, y)

        return output

    def predict(self, x):

        # Estimate using Torch Model
        t = torch.tensor(x, device=device, dtype=torch.float32)
        t = self.model(t)

        # Bring Torch Tensor from GPU to System Host Memory
        t = t.cpu()

        # Return Estimate as Numpy Float
        y = t.data.numpy()

        return y

    def eval(self, base_truth, meta_batch_size, gradient_steps, inner_step_size):

        # Sample Points from Task Sample Space
        x, y = sample_points(base_truth, self.meta_batch_size)

        # Model Base Estimate over Sample Space
        estimate = [self.predict(sample_space[:, None])]

        # Store Meta-Initialization Weights
        meta_weights = deepcopy(self.model.state_dict())

        # Get Estimate Loss over Meta-Initialization
        loss_t = self.loss(x, y).cpu()
        meta_loss = loss_t.data.numpy()

        # Calculcate Estimate over Gradient Steps
        for step in range(gradient_steps):

            # Calculate Evaluation Loss and Backpropagate
            eval_loss = self.loss(x, y)
            eval_loss.backward()

            # Update Model Estimate Parameters
            for theta in self.model.parameters():

                # Get Parameter Gradient
                grad = theta.grad.data

                # Update Model Parameter
                theta.data -= self.inner_step_size * grad

            # Update Estimate over Sample Space
            estimate.append(self.predict(sample_space[:, None]))

        # Get Estimate Loss over Evaluation
        loss_t = self.loss(x, y).cpu()
        estimate_loss = loss_t.data.numpy()
        evaluation_loss = abs(meta_loss - estimate_loss) / meta_batch_size

        # Restore Meta-Initialization Weights
        self.model.load_state_dict(meta_weights)

        return estimate, evaluation_loss
