import torch
import torch.nn as nn




""" Quant Noise"""
def quant_noise(x, bit, n_type="gaussian"):
    tensor = x.clone()
    flat = tensor.view(-1)
    scale = flat.max() - flat.min()
    unit = 1 / (2**bit - 1)

    if n_type == "uniform":
        noise_source = torch.rand_like(flat) - 0.5
    elif n_type == "gaussian":
        noise_source = torch.randn_like(flat) / 2

    noise = scale * unit * noise_source
    noisy = flat + noise
    return noisy.view_as(tensor).detach()


""" HWGQ """

hwgq_steps = {
    1: 1.0576462792297525,
    2: 0.6356366866203315,
    3: 0.3720645813370479,
    4: 0.21305606790772952,
    8: 0.020300567823662602,
    16: 9.714825915156693e-05,
}

class _hwgq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step):
        y = torch.round(x / step) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HWGQ(nn.Module):
    def __init__(self, bit=2, act=None):
        super(HWGQ, self).__init__()
        self.bit = bit
        if bit < 32:
            self.step = hwgq_steps[bit]
        elif bit == 32:
            self.prelu = act
            self.step = None
        else:
            raise NotImplementedError 

    def forward(self, x):
        if self.bit >= 32:
            if not self.act is None:
                x = self.act(x)
            return x
        lvls = float(2**self.bit - 1)
        clip_thr = self.step * lvls
        y = x.clamp(min=0.0, max=clip_thr)
        return _hwgq.apply(y, self.step)



"""  LSQ  """
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(nn.Module):
    def __init__(
        self, bit, all_positive=False, symmetric=False, per_channel=False, weight=None, act=None
    ):
        super(LsqQuan, self).__init__()
        self.bit = bit
        self.act = act
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2**bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = -(2 ** (bit - 1)) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = -(2 ** (bit - 1))
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))
        
        if weight is not None:
            self.init_from(weight)    

    def init_from(self, weight):
        if self.per_channel:
            self.s = nn.Parameter(
                weight.detach().abs().mean(dim=list(range(1, weight.dim())), keepdim=True)
                * 2
                / (self.thd_pos**0.5)
            )
        else:
            self.s = nn.Parameter(
                weight.detach().abs().mean() * 2 / (self.thd_pos**0.5)
            )

    def forward(self, x):
        if self.bit >= 32:
            if not self.act is None:
                x = self.act(x)
            return x

  
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            
        device = x.device
        s_scale = grad_scale(self.s, s_grad_scale).to(device)
        x = x / (s_scale)
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * (s_scale)
        return x
    
class Skip(nn.Module):
    def __init__(self):
        super(Skip, self).__init__()
    
    def forward(self,x):
        return x
        
