import math
import torch
from transformers import Adafactor, AdamW

# stochastic rounding for bfloat16
# The implementation was provided by 2kpr. Thank you very much!

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    # create a random 16 bit integer
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


@torch.no_grad()
def adafactor_step_param(self, p, group):
    if p.grad is None:
        return
    grad = p.grad
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()
    if grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients.")

    state = self.state[p]
    grad_shape = grad.shape

    factored, use_first_moment = Adafactor._get_options(group, grad_shape)
    # State Initialization
    if len(state) == 0:
        state["step"] = 0

        if use_first_moment:
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(grad)
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
            state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad)

        state["RMS"] = 0
    else:
        if use_first_moment:
            state["exp_avg"] = state["exp_avg"].to(grad)
        if factored:
            state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
            state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
        else:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

    p_data_fp32 = p
    if p.dtype in {torch.float16, torch.bfloat16}:
        p_data_fp32 = p_data_fp32.float()

    state["step"] += 1
    state["RMS"] = Adafactor._rms(p_data_fp32)
    lr = Adafactor._get_lr(group, state)

    beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
    update = (grad**2) + group["eps"][0]
    if factored:
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]

        exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
        exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

        # Approximation of exponential moving average of square of gradient
        update = Adafactor._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
        update.mul_(grad)
    else:
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
        update = exp_avg_sq.rsqrt().mul_(grad)

    update.div_((Adafactor._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
    update.mul_(lr)

    if use_first_moment:
        exp_avg = state["exp_avg"]
        exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
        update = exp_avg

    if group["weight_decay"] != 0:
        p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

    p_data_fp32.add_(-update)

    # if p.dtype in {torch.float16, torch.bfloat16}:
    #    p.copy_(p_data_fp32)

    if p.dtype == torch.bfloat16:
        copy_stochastic_(p, p_data_fp32)
    elif p.dtype == torch.float16:
        p.copy_(p_data_fp32)

@torch.no_grad()
def adamw_step_param(self, p, group):
    if p.grad is None:
        return
    grad = p.grad
    if grad.is_sparse:
        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

    state = self.state[p]

    # State initialization
    if len(state) == 0:
        state["step"] = 0
        # Exponential moving average of gradient values
        state["exp_avg"] = torch.zeros_like(p)
        # Exponential moving average of squared gradient values
        state["exp_avg_sq"] = torch.zeros_like(p)

    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
    beta1, beta2 = group["betas"]

    state["step"] += 1

    # Decay the first and second moment running average coefficient
    # In-place operations to update the averages at the same time
    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    denom = exp_avg_sq.sqrt().add_(group["eps"])

    step_size = group["lr"]
    # if group["correct_bias"]:  # No bias correction for Bert
    #     bias_correction1 = 1.0 - beta1 ** state["step"]
    #     bias_correction2 = 1.0 - beta2 ** state["step"]
    #     step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

    p.addcdiv_(exp_avg, denom, value=-step_size)

    # Just adding the square of the weights to the loss function is *not*
    # the correct way of using L2 regularization/weight decay with Adam,
    # since that will interact with the m and v parameters in strange ways.
    #
    # Instead we want to decay the weights in a manner that doesn't interact
    # with the m/v parameters. This is equivalent to adding the square
    # of the weights to the loss with plain (non-momentum) SGD.
    # Add weight decay at the end (fixed version)
    if group["weight_decay"] > 0.0:
        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))


@torch.no_grad()
def optimizer_step(self, optimizer_step_param, closure=None):
    """
    Performs a single optimization step

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group["params"]:
            optimizer_step_param(self, p, group)

    return loss

def patch_optimizer_fused(optimizer, optimizer_type):
    print(type(optimizer))
    if optimizer_type.lower()=='adamw':
        print("Using AdamW Fused")
        optimizer.step_param = adamw_step_param.__get__(optimizer)
        optimizer.step = optimizer_step.__get__(optimizer, adamw_step_param)
    if optimizer_type.lower()=='adafactor':
        print("Using Adafactor Fused")
        optimizer.step_param = adafactor_step_param.__get__(optimizer)
        optimizer.step = optimizer_step.__get__(optimizer, adafactor_step_param)

