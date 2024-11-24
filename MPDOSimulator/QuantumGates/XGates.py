"""
Author: weiguo_ma
Time: 11.24.2024
Contact: weiguo.m@iphy.ac.cn
"""

from typing import Union, Optional

from torch import Tensor, cos, sin, complex64
from torch import tensor as tensor_torch

from .AbstractGate import QuantumGate


class XGate(QuantumGate):
    """
    X gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(XGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'X'

    @property
    def tensor(self):
        return tensor_torch([[0, 1], [1, 0]], dtype=self.dtype, device=self.device)

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


class RXGate(QuantumGate):
    """
    RX gate.
    """

    def __init__(self, theta: Union[Tensor, float],
                 ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(RXGate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self.theta = self._check_Para_Tensor(theta)
        self.para = self.theta

    @property
    def name(self):
        return 'RX'

    @property
    def tensor(self):
        _tC, _tS = cos(self.theta / 2), sin(self.theta / 2)
        return tensor_torch(data=[[_tC, -1j * _tS], [-1j * _tS, _tC]], dtype=self.dtype, device=self.device)

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


class CXGate(QuantumGate):
    """
    CX gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(CXGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'CX'

    @property
    def tensor(self):
        return tensor_torch(
            data=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=self.dtype, device=self.device
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


class RXXGate(QuantumGate):
    """
    RXX gate.
    """

    def __init__(self, theta: Union[Tensor, float],
                 ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(RXXGate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self.theta = self._check_Para_Tensor(theta)
        self.para = self.theta

    @property
    def name(self):
        return 'RXX'

    @property
    def tensor(self):
        _tC, _tS = cos(self.theta / 2), sin(self.theta / 2)
        return tensor_torch(
            data=[[_tC, 0, 0, -1j * _tS], [0, _tC, -1j * _tS, 0], [0, -1j * _tS, _tC, 0], [-1j * _tS, 0, 0, _tC]],
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
        return True
