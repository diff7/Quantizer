import torch

import warnings
import torch.nn as nn

from QSB.qconfig import QConfig
from QSB.quantizers import  quant_noise


# TODO move quantizers params out


def get_normalized(alpha, margin=0):
    alphas = nn.functional.softmax(alpha, -1)
    alphas = alphas / (alphas.sum() + margin)
    return alphas

class FlopConv(nn.Conv2d):

    """
    Flops are computed for square kernel
    FLOPs = 2 x Cin x Cout x k**2 x Wout x Hout / groups
    We use 2 because 1 for multiplocation and 1 for addition
    Hout = Hin + 2*padding[0] - dilation[0] x (kernel[0]-1)-1
          --------------------------------------------------- + 1
                                stride
    Wout same as above
    NOTE: We do not account for bias term
    """

    def __init__(self, **kwargs):
        super(FlopConv, self).__init__(**kwargs)
        self.kernel = self.to_tuple(self.kernel_size)
        self.param_size = (
            2
            * self.in_channels
            * self.out_channels
            * self.kernel[0]
            * self.kernel[1]
            / self.groups
        )  # * 1e-6  # stil unsure why we use 1e-6
        self.register_buffer("flops", torch.tensor(0, dtype=torch.float))
        self.register_buffer("memory_size", torch.tensor(0, dtype=torch.float))

    def to_tuple(self, value):
        if type(value) == int:
            return (value, value)
        if type(value) == tuple:
            return value

    def forward(self, input_x, weights):
        """
        BATCH x C x W x H
        """
        # get the same device to avoid errors
        output = self._conv_forward(input_x, weights, bias=self.bias)

        w_out = output.shape[2]
        h_out = output.shape[3]

        device = input_x.device

        c_in, w_in, h_in = input_x.shape[1], input_x.shape[2], input_x.shape[3]

        tmp = torch.tensor(c_in * w_in * h_in, dtype=torch.float).to(device)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(
            self.param_size * w_out * h_out, dtype=torch.float
        ).to(device)
        self.flops.copy_(tmp)
        del tmp

        return output

    def _fetch_info(self):
        return self.flops.item(), self.memory_size.item()


class BaseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
        qconfig,
    ):

        super(BaseConv, self).__init__()
        self.qconfig = qconfig
        self.bits = qconfig.bits
        self.alphas = nn.Parameter(torch.ones(len(self.bits))/len(self.bits), requires_grad=True)
        self.conv = FlopConv(
             in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, input_x):
        raise NotImplementedError

    def _fetch_info(self):
        bit_ops, mem = 0, 0
        ops, m = self.conv._fetch_info()
        alphas = get_normalized(self.alphas)
        for bit, alpha in zip(self.bits, alphas):
            bit_ops += (alpha * ops * bit**2)/2
            mem += alpha * m * bit
        return bit_ops, mem


    def get_bit(self):
        idx = torch.argmax(self.alphas)
        bit = self.qconfig.bits[idx]
        return bit

    def get_weight(self):
        return self.conv.weight, self.conv.bias
    
    def _get_arch_values(self):
        return self.get_bit(), get_normalized(self.alphas)


class SingleConv(BaseConv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_q_fucntions(self, bit):
        self.act = self.qconfig.act_quantizer(bit, None)
        self.q_fn = self.qconfig.weight_quantizer(
            bit, self.conv.weight
        )
        self.bit = bit

    def forward(self, input_x):
        quantized_weight = self.q_fn(self.conv.weight)
        quantized_act = self.act(input_x)
        out = self.conv(quantized_act, quantized_weight)
        return out

    def fetch_info(self):
        f, m = self.conv._fetch_info()
        return (f * self.bit**2)/2, m * self.bit

    def set_weights(self, weight, bias):
        self.conv.weight = weight
        self.conv.bias = bias
        

    def get_bit(self):
        raise NotImplementedError

    def get_weight(self):
        raise NotImplementedError

    @classmethod
    def from_module(cls, mod: nn.Module, qconfig: QConfig = None):

        conv = cls(
            in_channels=mod.in_channels,
            out_channels= mod.out_channels,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )

        return conv


class QuaNoiseConv2d(BaseConv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fp_portion = 1 / (len(self.bits) +1)
        self.act = nn.PReLU()

    def forward(self, input_x):
        input_x = self.act(input_x)

        # keep FP portion of weights
        weights = self.fp_portion * self.conv.weight
        acts = self.fp_portion * input_x

        # rescale alphas among each other
        alphas = get_normalized(self.alphas, self.fp_portion)
        
        for alpha, bit in zip(alphas, self.bits):
            weights += alpha * quant_noise(self.conv.weight, bit)
            acts += alpha * quant_noise(input_x, bit)
        return self.conv(acts, weights)


class SharedQAConv2d(BaseConv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        weight_quantizer = self.qconfig.weight_quantizer
        act_quantizer = self.qconfig.act_quantizer

        self.acts = nn.ModuleList(
            [act_quantizer(bit, None) for bit in self.bits]
        )
        self.q_fn = nn.ModuleList(
            [weight_quantizer(bit, self.conv.weight) for bit in self.bits]
        )

    def forward(self, input_x):
        weights = torch.zeros_like(self.conv.weight)
        acts = torch.zeros_like(input_x)
        alphas = get_normalized(self.alphas)
        for alpha, act, q_fn in zip(alphas, self.acts, self.q_fn):
            weights += alpha * q_fn(self.conv.weight)
            acts += alpha * act(input_x)

        return self.conv(acts, weights)


class SearchConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        qconfig: QConfig = None,
    ):

        super(SearchConv2d, self).__init__()

        self.qconfig = qconfig
        if qconfig.noise_search:
            self.conv_func = QuaNoiseConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
                qconfig=qconfig,
            )

            warnings.warn("Using quantization noise.")

        else:
            self.conv_func = SharedQAConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
                qconfig=qconfig,
            )

            warnings.warn("Using shared weights.")

    def forward(self, input_x):
        return self.conv_func(input_x)

    # def set_alphas(self, alphas):
    #     for m in self.modules():
    #         if isinstance(m, self.conv_func):
    #             m.set_alphas(alphas)
    #     self.alphas = alphas
    
    def get_alphas(self):
        return self.conv_func.alphas

    def fetch_info(self):
        sum_flops = 0
        sum_memory = 0
        f, mem = self.conv_func._fetch_info()
        sum_flops += f
        sum_memory += mem

        return sum_flops, sum_memory

    @classmethod
    def from_module(cls, mod: nn.Module, qconfig: QConfig = None):

        conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )

        return conv

    def set_single_conv(self, bit=0, use_max=False):
        if use_max:
            bit = self.conv_func.get_bit()
        
        conv = SingleConv.from_module(self.conv_func.conv, self.qconfig)    
        weight, bias = self.conv_func.get_weight()
        conv.set_q_fucntions(bit)
        conv.set_weights(weight, bias)
        return conv
    
    def get_arch_values(self):
        return self.conv_func._get_arch_values()


if __name__ == "__main__":
    """
    Test functionality
    """

    
    class Model(nn.Module):
        def __init__(self, qconfig):
            super(Model, self).__init__()
            
            conv = nn.Conv2d(3,3, 5, stride=1, padding='same')
            self.conv = SearchConv2d.from_module(conv, qconfig)
            
        def  forward(self, x):
            return self.conv(x)
    
    input_x = torch.randn(1,3,10,10)
        
    qconfig = QConfig()
    print(qconfig)
    model = Model(qconfig)
    out = model(input_x)
    model.conv.set_single_conv()
    out = model(input_x)
    
    qconfig = QConfig(noise_search=True)
    print(qconfig)
    model = Model(qconfig)
    out = model(input_x)
    model.conv.set_single_conv()
    out = model(input_x)
    
    qconfig = QConfig(act_quantizer='HWGQ', weight_quantizer='HWGQ')
    print(qconfig)
    model = Model(qconfig)
    out = model(input_x)
    model.conv.set_single_conv()
    out = model(input_x)
    
    
         
        
    
