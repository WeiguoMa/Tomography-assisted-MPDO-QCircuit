"""
Author: weiguo_ma
Time: 11.24.2024
Contact: weiguo.m@iphy.ac.cn
"""

from typing import Union, Optional

import numpy as np
from torch import Tensor, exp, complex64
from torch import tensor as torch_tensor

from .AbstractGate import QuantumGate


class SGate(QuantumGate):
    """
    S gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(SGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'S'

    @property
    def tensor(self):
        return torch_tensor(data=[[1, 0], [0, 1j]], dtype=self.dtype, device=self.device)

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


class SDGGate(QuantumGate):
    """
    S gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(SDGGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'Sdg'

    @property
    def tensor(self):
        return torch_tensor(data=[[1, 0], [0, -1j]], dtype=self.dtype, device=self.device)

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


class TGate(QuantumGate):
    """
    T gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(TGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'T'

    @property
    def tensor(self):
        return torch_tensor(data=[[1, 0], [0, (1 + 1j) / np.sqrt(2)]], dtype=self.dtype, device=self.device)

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


class PGate(QuantumGate):
    """
    P gate.
    """

    def __init__(self, theta: Union[Tensor, float],
                 ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(PGate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self.theta = self._check_Para_Tensor(theta)
        self.para = self.theta

    @property
    def name(self):
        return 'P'

    @property
    def tensor(self):
        return torch_tensor(
            data=[[1, 0], [0, exp(self.theta * 1j)]],
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


class CPGate(QuantumGate):
    """
    CP gate.
    """

    def __init__(self, theta: Union[Tensor, float],
                 ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(CPGate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self.theta = self._check_Para_Tensor(theta)
        self.para = self.theta

    @property
    def name(self):
        return 'CP'

    @property
    def tensor(self):
        return torch_tensor(
            data=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, exp(1j * self.theta)]],
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
