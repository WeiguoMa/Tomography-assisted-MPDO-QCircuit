"""
Author: weiguo_ma
Time: 11.24.2024
Contact: weiguo.m@iphy.ac.cn
"""

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor, nn, complex64, tensor


class QuantumGate(ABC, nn.Module):
    """
    Base class for quantum gates.
    """

    def __init__(self, ideal: Optional[bool] = True, dtype=complex64, device: str = 'cpu'):
        super(QuantumGate, self).__init__()
        self.para = None
        self._ideal = ideal
        self.dtype, self.device = dtype, device

    def _check_Para_Tensor(self, *parameters):
        def __convert(param):
            if isinstance(param, Tensor):
                return param.to(dtype=self.dtype, device=self.device)
            elif isinstance(param, float):
                return tensor(param, dtype=self.dtype, device=self.device)
            raise ValueError(f"Invalid type for gate parameter: {type(param)}")

        params = [__convert(param) for param in parameters]
        return params if len(parameters) > 1 else params[0]

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def tensor(self):
        pass

    @property
    @abstractmethod
    def rank(self):
        pass

    @property
    @abstractmethod
    def dimension(self):
        pass

    @property
    @abstractmethod
    def single(self) -> bool:
        pass

    @property
    @abstractmethod
    def variational(self) -> bool:
        pass

    @property
    def para(self):
        return self._para

    @para.setter
    def para(self, para: Tensor):
        self._para = para

    @property
    def ideal(self) -> Optional[bool]:
        return self._ideal

    @ideal.setter
    def ideal(self, value: Optional[bool]):
        if not isinstance(value, bool):
            raise ValueError("Value must be bool.")
        self._ideal = value
