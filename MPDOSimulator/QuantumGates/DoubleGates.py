"""
Author: weiguo_ma
Time: 11.24.2024
Contact: weiguo.m@iphy.ac.cn
"""
from typing import Union, Optional
from warnings import warn

from torch import Tensor, complex64
from torch import tensor as torch_tensor

from .AbstractGate import QuantumGate


class IIGate(QuantumGate):
    """
    II gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(IIGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'II'

    @property
    def tensor(self):
        return torch_tensor(
            data=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=self.dtype, device=self.device
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


class CNOTGate(QuantumGate):
    """
    CNOT gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(CNOTGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'CNOT'

    @property
    def tensor(self):
        return torch_tensor(
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


class ISWAPGate(QuantumGate):
    """
    iSWAP gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(ISWAPGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'ISWAP'

    @property
    def tensor(self):
        return torch_tensor(
            data=[[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=self.dtype, device=self.device
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


class SWAPGate(QuantumGate):
    """
    SWAP gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(SWAPGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'SWAP'

    @property
    def tensor(self):
        return torch_tensor(
            data=[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=self.dtype, device=self.device
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


class ArbDoubleGate(QuantumGate):
    """
    Arbitrary two-qubit gate.
    """

    def __init__(self, matrix: Tensor, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(ArbDoubleGate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self._matrix = matrix.to(dtype=self.dtype, device=self.device)

    @property
    def name(self):
        return 'ArbDoubleGate'

    @property
    def tensor(self):
        if self._matrix.shape != (2, 2, 2, 2):
            warn('You are probably adding a noisy double qubit gate, current shape is {}'.format(self._matrix.shape))
        return self._matrix.reshape(2, 2, 2, 2, -1).squeeze()

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
