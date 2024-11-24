"""
Author: weiguo_ma
Time: 11.24.2024
Contact: weiguo.m@iphy.ac.cn
"""

from typing import Union, Optional
from warnings import warn

import numpy as np
from torch import Tensor, complex64, exp, cos, sin
from torch import tensor as torch_tensor

from .AbstractGate import QuantumGate


class IGate(QuantumGate):
    """
    I gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(IGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'I'

    @property
    def tensor(self):
        return torch_tensor(data=[[1, 0], [0, 1]], dtype=self.dtype, device=self.device)

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


class HGate(QuantumGate):
    """
    H gate.
    """

    def __init__(self, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(HGate, self).__init__(ideal=ideal, dtype=dtype, device=device)

    @property
    def name(self):
        return 'H'

    @property
    def tensor(self):
        return torch_tensor(data=[[1, 1], [1, -1]], dtype=self.dtype, device=self.device) / np.sqrt(2)

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


class U1Gate(QuantumGate):
    """
    U1 gate.
    """

    def __init__(self, theta: Union[Tensor, float],
                 ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(U1Gate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self.theta = self._check_Para_Tensor(theta)
        self.para = self.theta

    @property
    def name(self):
        return 'U1'

    @property
    def tensor(self):
        return torch_tensor(data=[[1, 0], [0, exp(1j * self.theta)]], dtype=self.dtype, device=self.device)

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


class U2Gate(QuantumGate):
    """
    U2 gate.
    """

    def __init__(self, phi: Union[Tensor, float], lam: Union[Tensor, float],
                 ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(U2Gate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self.phi, self.lam = self._check_Para_Tensor(phi, lam)
        self.para = [self.phi, self.lam]

    @property
    def name(self):
        return 'U2'

    @property
    def tensor(self):
        _lM, _pM = exp(1j * self._lam), exp(1j * self.phi)
        return torch_tensor(data=[[1, -_lM], [_pM, _lM * _pM]], device=self.device, dtype=self.dtype) / np.sqrt(2)

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


class U3Gate(QuantumGate):
    """
    U3 gate.
    """

    def __init__(self, theta: Union[Tensor, float], phi: Union[Tensor, float], lam: Union[Tensor, float],
                 ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(U3Gate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self.theta, self.phi, self.lam = self._check_Para_Tensor(theta, phi, lam)
        self.para = [self.theta, self.phi, self.lam]

    @property
    def name(self):
        return 'U3'

    @property
    def tensor(self):
        _lM, _pM = exp(1j * self.lam), exp(1j * self.phi)
        _tC, _tS = cos(self.theta / 2), sin(self.theta / 2)
        return torch_tensor(data=[[_tC, -_lM * _tS], [_pM * _tS, _lM * _pM * _tC]], device=self.device,
                            dtype=self.dtype)

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


class ArbSingleGate(QuantumGate):
    """
    ArbSingleGate gate.
    """

    def __init__(self, tensor: Tensor, ideal: Optional[bool] = None, dtype=complex64, device: Union[str, int] = 'cpu'):
        super(ArbSingleGate, self).__init__(ideal=ideal, dtype=dtype, device=device)
        self._matrix = tensor.to(dtype=self.dtype, device=self.device)

    @property
    def name(self):
        return 'ArbSingleGate'

    @property
    def tensor(self):
        if self._matrix.shape != (2, 2):
            warn('You are probably adding a noisy single qubit gate, current shape is {}'.format(self._matrix.shape))
        return self._matrix.reshape(2, 2, -1).squeeze()

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
