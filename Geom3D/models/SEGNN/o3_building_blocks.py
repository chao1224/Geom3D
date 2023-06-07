import torch
import torch.nn as nn
from e3nn.nn import Gate
from e3nn.o3 import (FullyConnectedTensorProduct, Irreps, Linear,
                     spherical_harmonics)


class O3TensorProduct(torch.nn.Module):
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None, tp_rescale=True) -> None:
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_out = irreps_out
        # Init irreps_in2
        if irreps_in2 == None:
            self.irreps_in2_provided = False
            self.irreps_in2 = Irreps("1x0e")
        else:
            self.irreps_in2_provided = True
            self.irreps_in2 = irreps_in2
        self.tp_rescale = tp_rescale

        # Build the layers
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out, shared_weights=True, normalization='component')

        # For each zeroth order output irrep we need a bias
        # So first determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()
        # Store tuples of slices and corresponding biaes in a list
        self.biases = []
        self.biases_slices = []
        self.biases_slice_idx = []
        for slice_idx in range(len(self.irreps_out_orders)):
            if self.irreps_out_orders[slice_idx] == 0:
                out_slice = irreps_out.slices()[slice_idx]
                out_bias = torch.nn.Parameter(
                    torch.zeros(self.irreps_out_dims[slice_idx], dtype=self.tp.weight.dtype))
                self.biases += [out_bias]
                self.biases_slices += [out_slice]
                self.biases_slice_idx += [slice_idx]
        self.biases = torch.nn.ParameterList(self.biases)

        # Initialize the correction factors
        self.slices_sqrt_k = {}

        # Initialize similar to the torch.nn.Linear
        self.tensor_product_init()

    def tensor_product_init(self) -> None:
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                slice_idx = instr[2]
                mul_1, mul_2, mul_out = weight.shape
                fan_in = mul_1 * mul_2
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                            fan_in if slice_idx in slices_fan_in.keys() else fan_in)

            # Do the initialization of the weights in each instruction
            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                slice_idx = instr[2]
                if self.tp_rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.
                weight.data.uniform_(-sqrt_k, sqrt_k)
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            for (out_slice_idx, out_slice, out_bias) in zip(self.biases_slice_idx, self.biases_slices, self.biases):
                sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
                out_bias.uniform_(-sqrt_k, sqrt_k)

    def forward_tp_rescale_bias(self, data_in1, data_in2=None) -> torch.Tensor:
        if data_in2 == None:
            data_in2 = torch.ones_like(data_in1[:, 0:1])

        data_out = self.tp(data_in1, data_in2)
        # Apply corrections
        if self.tp_rescale:
            for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
                data_out[:, slice] /= slice_sqrt_k
        # Add the biases
        for (_, slice, bias) in zip(self.biases_slice_idx, self.biases_slices, self.biases):
            data_out[:, slice] += bias
        # Return result
        return data_out

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        return data_out


class O3TensorProductSwishGate(O3TensorProduct):
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None) -> None:
        # For the gate the output of the linear needs to have an extra number of scalar irreps equal to the amount of
        # non scalar irreps:
        # The first type is assumed to be scalar and passed through the activation
        irreps_g_scalars = Irreps(str(irreps_out[0]))
        # The remaining types are gated
        irreps_g_gate = Irreps("{}x0e".format(irreps_out.num_irreps - irreps_g_scalars.num_irreps))
        irreps_g_gated = Irreps(str(irreps_out[1:]))
        # So the gate needs the following irrep as input, this is the output irrep of the tensor product
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()

        # Build the layers
        super(O3TensorProductSwishGate, self).__init__(irreps_in1, irreps_g, irreps_in2)
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(irreps_g_scalars, [nn.SiLU()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            self.gate = nn.SiLU()

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        # Apply the gate
        data_out = self.gate(data_out)
        # Return result
        return data_out
