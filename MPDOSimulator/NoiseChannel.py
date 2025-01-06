"""
Author: weiguo_ma
Time: 04.17.2023
Contact: weiguo.m@iphy.ac.cn
"""
import warnings
from typing import Optional, Union

import numpy as np
import tensornetwork as tn
import torch as tc

from .ChipInfo import ChipInformation
from .Tools import select_device

tn.set_default_backend("pytorch")


class NoiseChannel:
    def __init__(self, chip: Optional[str] = None, dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        self.dtype = dtype
        self.device = select_device(device)

        # chip Info
        self.chip = (ChipInformation().__getattribute__(chip or 'worst'))()
        self.bath_rate = self.chip.bath_rate
        self.decay_rate = self.chip.decay_rate
        self.dephasing_rate = self.chip.dephasing_rate
        self.T1 = self.chip.T1
        self.T2 = self.chip.T2
        self.GateTime = self.chip.gateTime
        self.dpc_errorRate = self.chip.dpc_errorRate

        self._basisPauli = [tc.tensor([[1, 0], [0, 1]], dtype=dtype, device=device),
                            tc.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
                            tc.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
                            tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)]

        # Noise channels' tensor
        self.decayTensor = self.decay(self.decay_rate, self.GateTime)
        # self.decayTensor2 = self.kron_tensor(self.decayTensor)
        self.dephasingTensor = self.dephasing(self.dephasing_rate, self.GateTime)
        # self.dephasingTensor2 = self.kron_tensor(self.dephasingTensor)
        self.dpCTensor = self.depolarization_noise_channel(p=self.dpc_errorRate)
        self.dpCTensor2 = self.depolarization_noise_channel(p=self.dpc_errorRate, qn=2)
        self.apdeCTensor = self.amp_phase_damping_error(time=self.GateTime, T1=self.T1, T2=self.T2)

    # def kron_tensor(self, required_tensor: tc.Tensor):
    #     return tc.einsum('ij, klm -> ikjlm', self._basisPauli[0], required_tensor)

    def depolarization_noise_channel(self, p: float, qn: int = 1) -> tc.Tensor:
        r"""
        Constructs the depolarization noise channel tensor.

        Parameters:
        p : float
            The probability of the depolarization noise.
        qn : int, optional
            The number of qubits, default is 1.

        Returns:
        tc.Tensor
            The tensor representing the depolarization noise channel.

        Raises:
        ValueError
            If the probability 'p' is not in the range [0, 1].

        Notes:
        The depolarization channel is modeled as:
            \epsilon(\rho) = (1 - 3p/4)\rho + p/4(X\rhoX + Y\rhoY + Z\rhoZ) for a single qubit.
        """
        if not 0 <= p <= 1:
            raise ValueError('Probability p must be in the range [0, 1].')

        error_prob_list = [np.sqrt(1 - (4 ** qn - 1) * p / (4 ** qn))]
        error_prob_list.extend([np.sqrt(p / (4 ** qn))] * (4 ** qn - 1))

        error_diag = tc.diag(tc.tensor(error_prob_list, dtype=self.dtype, device=self.device))

        # Direct implementation of tensor product using torch.kron
        def iter_kron_product(input_list, remaining_qubits):
            if remaining_qubits == 0:
                return input_list
            output_list = [tc.kron(e, b) for e in input_list for b in self._basisPauli]
            return iter_kron_product(output_list, remaining_qubits - 1)

        dpc_tensor = tc.stack(iter_kron_product(self._basisPauli, qn - 1)).to(device=self.device)
        dpc_tensor = tc.einsum('ij, jfk -> fki', error_diag, dpc_tensor)
        dpc_tensor = dpc_tensor.reshape([2] * (2 * qn) + [dpc_tensor.shape[-1]])
        return dpc_tensor

    def amp_phase_damping_error(self, time: float, T1: float, T2: float) -> tc.Tensor:
        r"""
        The amplitude-phase damping error.

        Args:
            time: The time of the amplitude-phase damping error;
            T1: The T1 time;
            T2: The T2 time.

        Returns:
            _apdc_tensor: The amplitude-phase damping error.

        Additional information:
            .. math::

                \epsilon(\rho) = \left(\begin{array}{cc}
                1 & 0 \\
                0 & \sqrt{1 - \frac{1}{2}e^{-\frac{t}{T_1}} - \frac{1}{2}e^{-\frac{t}{T_2}}} \\
                \end{array}\right)
                \left(\begin{array}{cc}
                1 & 0 \\
                0 & \sqrt{\frac{1}{2}e^{-\frac{t}{T_1}}} \\
                \end{array}\right)
                \left(\begin{array}{cc}
                1 & 0 \\
                0 & \sqrt{\frac{1}{2}e^{-\frac{t}{T_2}}} \\
                \end{array}\right)

        Attention:
            Time t should be as a working time in whole process, but not just a gate time.
            Time unit should be formed as nanoseconds: 'ns' uniformly.
        """
        if time < 0 or T1 <= 0 or T2 <= 0:
            raise ValueError('The time, T1, T2 must be greater than or equal to 0, '
                             'for some special cases time = 0 is allowed.')
        if time == 0:
            warnings.warn('The time is 0, which means the noise is not applied.')

        _T2p = 2 * T1 * T2 / (2 * T1 - T2)  # 1/T2 = 1/T2p + 1/(2T1)
        _param_T1 = 1 - np.exp(- time / T1)
        _param_T2 = 1 - np.exp(- time / _T2p)

        _apdc_tensor = tc.tensor(
            [
                [[1, 0], [0, np.sqrt(1 - (_param_T1 + _param_T2))]],
                [[0, 0], [0, np.sqrt(_param_T2)]],
                [[0, np.sqrt(_param_T1)], [0, 0]]
            ], dtype=self.dtype, device=self.device)
        _apdc_tensor = tc.einsum('ijk -> jki', _apdc_tensor)
        return _apdc_tensor

    def decay(self, gamma: float, gate_time: float):
        if gate_time <= 0:
            raise ValueError('The gate time must be greater than 0, or set the gate to IDEAL.')

        _param = 1 - np.exp(- self.bath_rate * gamma * gate_time)
        _kraus_tensor = tc.tensor(
            [
                [[1, 0], [0, np.sqrt(1 - _param)]],     # K_1
                [[0, np.sqrt(_param)], [0, 0]]          # K_2
            ], dtype=self.dtype, device=self.device
        )
        return _kraus_tensor.permute((1, 2, 0))

    def dephasing(self, gamma: float, gate_time: float):
        if gate_time <= 0:
            raise ValueError('The gate time must be greater than 0, or set the gate to IDEAL.')

        _param = 1 - np.exp(- self.bath_rate * gamma * gate_time)
        _sqrt_p, _sqrt_1_p = np.sqrt(_param), np.sqrt(1 - _param)
        _kraus_tensor = tc.tensor(
            [
                [[_sqrt_1_p, 0], [0, _sqrt_1_p]],       # K_1
                [[_sqrt_p, 0], [0, 0]],                 # K_2
                [[0, 0], [0, _sqrt_p]]                  # K_3
            ], dtype=self.dtype, device=self.device
        )
        return _kraus_tensor.permute((1, 2, 0))