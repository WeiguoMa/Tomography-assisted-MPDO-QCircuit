"""
Author: weiguo_ma
Time: 11.24.2024
Contact: weiguo.m@iphy.ac.cn
"""

from typing import Union, Optional

from torch import Tensor, complex64, cos, sin
from torch import tensor as torch_tensor

from .AbstractGate import QuantumGate


class YGate(QuantumGate):
    """
    Y gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(YGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'Y'

    @property
    def tensor(self):
        return torch_tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device)

    @property
    def rank(self):
        return 2

    @property
    def dimension(self):
        return [2, 2]

    @property
    def single(self) -> bool:
        return True

    @property
    def variational(self) -> bool:
        return False


class RYGate(QuantumGate):
    """
    RY gate.
    """

    def __init__(self, theta: Union[Tensor, float],
                 ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(RYGate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self.theta = self._check_Para_Tensor(theta)
        self.para = self.theta

    @property
    def name(self):
        return 'RY'

    @property
    def tensor(self):
        _tC, _tS = cos(self.theta / 2), sin(self.theta / 2)
        return torch_tensor(
            data=[[_tC, -_tS], [_tS, _tC]],
            dtype=self.dtype, device=self.device, requires_grad=self.theta.requires_grad
        )

    @property
    def rank(self):
        return 2

    @property
    def dimension(self):
        return [2, 2]

    @property
    def single(self) -> bool:
        return True

    @property
    def variational(self) -> bool:
        return True


class CYGate(QuantumGate):
    """
    CY gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(CYGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'CY'

    @property
    def tensor(self):
        return torch_tensor(
            data=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            dtype=self.dtype, device=self.device
        ).reshape((2, 2, 2, 2))

    @property
    def rank(self):
        return 4

    @property
    def dimension(self):
        return [[2, 2], [2, 2]]

    @property
    def single(self) -> bool:
        return False

    @property
    def variational(self) -> bool:
        return False


class RYYGate(QuantumGate):
    """
    RYY gate.
    """

    def __init__(self, theta: Union[Tensor, float], ideal: Optional[bool] = None, dtype=complex64,
                 device: Union[str, int] = 'cpu'):
        super(RYYGate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self.theta = self._check_Para_Tensor(theta)
        self.para = theta

    @property
    def name(self):
        return 'RYY'

    @property
    def tensor(self):
        _tC, _tS = cos(self.theta / 2), sin(self.theta / 2)
        return torch_tensor(
            data=[[_tC, 0, 0, -_tS], [0, _tC, _tS, 0], [0, -_tS, _tC, 0], [_tS, 0, 0, _tC]],
            dtype=self.dtype, device=self.device, requires_grad=self.theta.requires_grad
        ).reshape((2, 2, 2, 2))

    @property
    def rank(self):
        return 4

    @property
    def dimension(self):
        return [[2, 2], [2, 2]]

    @property
    def single(self) -> bool:
        return False

    @property
    def variational(self) -> bool:
        return True
