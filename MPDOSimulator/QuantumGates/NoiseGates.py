"""
Author: weiguo_ma
Time: 11.24.2024
Contact: weiguo.m@iphy.ac.cn
"""
import warnings
from typing import Union, Optional

from torch import Tensor, complex64

from .AbstractGate import QuantumGate
from ..RealNoise import czExp_channel, cpExp_channel


class CZEXPGate(QuantumGate):
    """
    CZ_EXP gate.
    """

    def __init__(self, tensor: Optional[Tensor] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(CZEXPGate, self).__init__(dtype=dtype, device=device)
        self.ideal = False
        self._matrix = tensor

    @property
    def name(self):
        return 'CZEXP'

    @property
    def tensor(self):
        # NO input may cause high memory cost and time cost
        if self._matrix is None:
            warnings.warn('No (sufficient) CZ input files, use default tensor.')
            return czExp_channel(filename='MPDOSimulator/chi/czDefault.mat').to(dtype=self.dtype, device=self.device)
        else:
            return self._matrix.to(dtype=self.dtype, device=self.device)

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


class CPEXPGate(QuantumGate):
    """
    CP_EXP gate.
    """

    def __init__(self, tensor: Optional[Tensor] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(CPEXPGate, self).__init__(dtype=dtype, device=device)
        self.ideal = False
        self._matrix = tensor

    @property
    def name(self):
        return 'CPEXP'

    @property
    def tensor(self):
        # NO input may cause high memory cost and time cost
        if self._matrix is None:
            warnings.warn('No (sufficient) CP input files, use default tensor.')
            return cpExp_channel().to(dtype=self.dtype, device=self.device)
        else:
            return self._matrix.to(dtype=self.dtype, device=self.device)

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
