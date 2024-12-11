"""
Author: weiguo_ma
Time: 11.27.2023
Contact: weiguo.m@iphy.ac.cn
"""
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any

from tensornetwork import AbstractNode, Node
from torch import Tensor, nn, pi, tensor, complex64, sqrt

from .RealNoise import czExp_channel, cpExp_channel
from .Tools import select_device


class QuantumCircuit(ABC, nn.Module):
    """
    Abstract class for quantum circuit.
    """

    def __init__(self,
                 realNoise: bool,
                 noiseFiles: Optional[Dict[str, Dict[str, Any]]] = None,
                 chi: Optional[int] = None, kappa: Optional[int] = None,
                 tnn_optimize: bool = True,
                 dtype=complex64,
                 device: Union[str, int] = 'cpu'):
        super(QuantumCircuit, self).__init__()

        self.device = select_device(device)
        self.dtype = dtype
        self.Truncate = None
        self.chi = chi
        self.kappa = kappa
        self.tnn_optimize = tnn_optimize

        self.layers = nn.Sequential()
        self._oqs_list = []

        # Noise
        self.realNoise = realNoise
        self.noiseFiles = noiseFiles if realNoise else None

        # Density Matrix
        self._initState = None
        self._vector = None
        self._stateNodes, self._dm, self._dmNodes = None, None, None
        self._samples, self._counts = None, None

        #
        if self.realNoise:
            self._load_exp_tensors()

        self._sequence, self._sequenceT = 0, 0

    def _load_exp_tensors(self):
        self._cz_expTensors, self._cp_expTensors = {}, {}
        if 'CZ' in self.noiseFiles.keys():
            _czDict = self.noiseFiles['CZ']
            for _keys in _czDict.keys():
                self._cz_expTensors[_keys] = czExp_channel(filename=_czDict[_keys]).to(self.device, dtype=self.dtype)
        if 'CP' in self.noiseFiles.keys():
            _cpDict = self.noiseFiles['CP']
            for _keys in _cpDict.keys():
                self._cp_expTensors[_keys] = cpExp_channel(filename=_cpDict[_keys]).to(self.device, dtype=self.dtype)

    def _load_projectors(self):
        _coefficient = 1 / sqrt(tensor(2, dtype=self.dtype, device=self.device))

        self._projectors_string = ['X', 'Y', 'Z']
        self._projectors = {
            0: [Node(
                tensor([1, 0], dtype=self.dtype, device=self.device), axis_names=['proj'], name='proj_2_0'
            )
                for _ in range(2)],
            1: [Node(
                tensor([0, 1], dtype=self.dtype, device=self.device), axis_names=['proj'], name='proj_2_1'
            )
                for _ in range(2)]
        }

    @property
    def dm(self):
        return self._dm

    @property
    def samples(self):
        return self._samples

    @property
    def counts(self):
        return self._counts

    @property
    def initial_state(self):
        return self._initState

    @property
    def stateNodes(self):
        return self._stateNodes

    @property
    def dmNodes(self):
        return self._dmNodes

    @property
    def vector(self):
        return self._vector

    @abstractmethod
    def cal_vector(self):
        pass

    @abstractmethod
    def cal_dm(self):
        pass

    @abstractmethod
    def evolve(self, state: List[AbstractNode]):
        pass

    def _add_module(self, _gate: nn.Module, oqs: List, headline: str):
        self._oqs_list.append(oqs)
        self.layers.add_module(headline + f'-S{self._sequence}', _gate)
        self._sequence += 1

    def _iter_add_module(self, _gate_list: List, oqs_list: List, _transpile: bool = False):
        for _gate, _oq in zip(_gate_list, oqs_list):
            if _transpile and _gate.name == 'CX' or 'CNOT':
                if _gate.para is None:
                    _headline = f"{_gate.name}{_oq}|None-TRANS"
                else:
                    _para_info = _gate.para.item() if isinstance(_gate.para, Tensor) else _gate.para
                    _headline = f"{_gate.name}{_oq}|({_para_info:.3f})-TRANS".replace('.', ';')

                self._add_module(_gate, _oq, _headline)
            else:
                self.__getattribute__(f'{_gate.name.lower()}')(_oq)

    def i(self, oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.SingleGates import IGate

        _headline = f"I{oqs}|None"

        self._add_module(IGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def h(self, oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.SingleGates import HGate

        _headline = f"H{oqs}|None"

        self._add_module(HGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def x(self, oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.XGates import XGate

        _headline = f"X{oqs}|None"

        self._add_module(XGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def y(self, oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.YGates import YGate

        _headline = f"Y{oqs}|None"

        self._add_module(YGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def z(self, oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.ZGates import ZGate

        _headline = f"Z{oqs}|None"

        self._add_module(ZGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def rx(self, theta: Union[Tensor, float], oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.XGates import RXGate

        _para_info = theta.item() if isinstance(theta, Tensor) else theta
        _headline = f"RX{oqs}|({_para_info:.3f})".replace('.', ';')

        self._add_module(RXGate(theta, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def ry(self, theta: Union[Tensor, float], oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.YGates import RYGate

        _para_info = theta.item() if isinstance(theta, Tensor) else theta
        _headline = f"RY{oqs}|({_para_info:.3f})".replace('.', ';')

        self._add_module(RYGate(theta, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def rz(self, theta: Union[Tensor, float], oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.ZGates import RZGate

        _para_info = theta.item() if isinstance(theta, Tensor) else theta
        _headline = f"RZ{oqs}|({_para_info:.3f})".replace('.', ';')

        self._add_module(RZGate(theta, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def rxx(self, theta: Union[Tensor, float], oq0: int, oq1: int, _ideal: Optional[bool] = None):
        oqs = [oq0, oq1]
        if not self.realNoise or _ideal:
            from .QuantumGates.XGates import RXXGate

            _para_info = theta.item() if isinstance(theta, Tensor) else theta
            _headline = f"RXX{oqs}|({_para_info:.3f})".replace('.', ';')

            self._add_module(RXXGate(theta, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)
        else:
            self.h(oqs, True)
            self.cx(oq0, oq1)
            self.rz(theta, oq1, True)
            self.cx(oq0, oq1)
            self.h(oqs, True)

    def ryy(self, theta: Union[Tensor, float], oq0: int, oq1: int, _ideal: Optional[bool] = None):
        oqs = [oq0, oq1]
        if not self.realNoise or _ideal:
            from .QuantumGates.YGates import RYYGate

            _para_info = theta.item() if isinstance(theta, Tensor) else theta
            _headline = f"RYY{oqs}|({_para_info:.3f})".replace('.', ';')

            self._add_module(RYYGate(theta, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)
        else:
            self.rx(tensor(pi / 2), oqs, True)
            self.cx(oq0, oq1)
            self.rz(theta, oq1, True)
            self.cx(oq0, oq1)
            self.rx(-tensor(pi / 2), oqs, True)

    def rzz(self, theta: Union[Tensor, float], oq0: int, oq1: int, _ideal: Optional[bool] = None):
        oqs = [oq0, oq1]
        if not self.realNoise or _ideal:
            from .QuantumGates.ZGates import RZZGate

            _para_info = theta.item() if isinstance(theta, Tensor) else theta
            _headline = f"RZZ{oqs}|({_para_info:.3f})".replace('.', ';')

            self._add_module(RZZGate(theta, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)
        else:
            self.cx(oq0, oq1)
            self.rz(theta, oq1, True)
            self.cx(oq0, oq1)

    def cx(self, oq0: int, oq1: int, _ideal: Optional[bool] = None):
        oqs = [oq0, oq1]
        if not self.realNoise or _ideal:
            from .QuantumGates.XGates import CXGate

            _headline = f"CX{oqs}|None"
            self._add_module(CXGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)
        else:
            self.ry(-tensor(pi / 2), oq1, True)
            self.cz(oq0, oq1)
            self.ry(tensor(pi / 2), oq1, True)

    def cy(self, oq0: int, oq1: int, _ideal: Optional[bool] = None):
        oqs = [oq0, oq1]
        if not self.realNoise or _ideal:
            from .QuantumGates.YGates import CYGate

            _headline = f"CY{oqs}|None"
            self._add_module(CYGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)
        else:
            raise NotImplementedError("EXPCYGate is not implemented yet.")

    def cz(self, oq0: int, oq1: int, _ideal: Optional[bool] = None):
        oqs = [oq0, oq1]
        if not self.realNoise or _ideal:
            from .QuantumGates.ZGates import CZGate

            _headline = f"CZ{oqs}|None"

            self._add_module(CZGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)
        else:
            from .QuantumGates.NoiseGates import CZEXPGate

            _headline = f"CZEXP{oqs}|None"

            _tensor = self._cz_expTensors.get(f'{oq0}{oq1}')
            if _tensor is None:
                _tensor = self._cz_expTensors.get(f'{oq1}{oq0}')

            self._add_module(CZEXPGate(_tensor, dtype=self.dtype, device=self.device), oqs, _headline)

    def cnot(self, oq0: int, oq1: int, _ideal: Optional[bool] = None):
        oqs = [oq0, oq1]
        if not self.realNoise or _ideal:
            from .QuantumGates.DoubleGates import CNOTGate

            _headline = f"CNOT{oqs}|None"
            self._add_module(CNOTGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)
        else:
            self.ry(-tensor(pi / 2), oq1, True)
            self.cz(oq0, oq1)
            self.ry(tensor(pi / 2), oq1, True)

    def swap(self, oq0: int, oq1: int, _ideal: Optional[bool] = None):
        oqs = [oq0, oq1]
        from .QuantumGates.DoubleGates import SWAPGate

        _headline = f"SWAP{oqs}|None"
        self._add_module(SWAPGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def iswap(self, oq0: int, oq1: int, _ideal: Optional[bool] = None):
        oqs = [oq0, oq1]
        from .QuantumGates.DoubleGates import ISWAPGate

        _headline = f"ISWAP{oqs}|None"
        self._add_module(ISWAPGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def s(self, oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.PhaseGates import SGate

        _headline = f"S{oqs}|None"

        self._add_module(SGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def t(self, oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.PhaseGates import TGate

        _headline = f"T{oqs}|None"

        self._add_module(TGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def p(self, theta: Union[Tensor, float], oqs: Union[List, int], _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.PhaseGates import PGate

        _para_info = theta.item() if isinstance(theta, Tensor) else theta
        _headline = f"P{oqs}|({_para_info:.3f})".replace('.', ';')

        self._add_module(PGate(theta, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def cp(self, theta: Optional[Union[Tensor, float]], oq0: int, oq1: int, _ideal: Optional[bool] = None):
        oqs = [oq0, oq1]

        if not self.realNoise or _ideal:
            from .QuantumGates.PhaseGates import CPGate

            _para_info = theta.item() if isinstance(theta, Tensor) else theta
            _headline = f"CP{oqs}|({_para_info:.3f})".replace('.', ';')

            self._add_module(CPGate(theta, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)
        else:
            from .QuantumGates.NoiseGates import CPEXPGate

            _headline = f"CPEXP{oqs}|None"

            _tensor = self._cp_expTensors.get(f'{oq0}{oq1}')
            if _tensor is None:
                _tensor = self._cp_expTensors.get(f'{oq1}{oq0}')
            self._add_module(CPEXPGate(_tensor, dtype=self.dtype, device=self.device), oqs, _headline)

    def arbSingle(self, data: Tensor, oqs: List, _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.SingleGates import ArbSingleGate

        _headline = f"ArbS{oqs}|None"

        self._add_module(ArbSingleGate(data, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def arbDouble(self, data: Tensor, oq1: int, oq2: int, _ideal: Optional[bool] = None):
        oqs = [oq1, oq2]
        from .QuantumGates.DoubleGates import ArbDoubleGate

        _headline = f"ArbD{oqs}|None"

        self._add_module(ArbDoubleGate(data, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def ii(self, oq1: int, oq2: int, _ideal: Optional[bool] = None):
        oqs = [oq1, oq2]
        from .QuantumGates.DoubleGates import IIGate

        _headline = f"II{oqs}|None"

        self._add_module(IIGate(_ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def u1(self, theta: Union[Tensor, float], oqs: List, _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.SingleGates import U1Gate

        _para_info = theta.item() if isinstance(theta, Tensor) else theta
        _headline = f"U1{oqs}|({_para_info:.3f})".replace('.', ';')

        self._add_module(U1Gate(theta, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def u2(self, phi: Union[Tensor, float], lam: Union[Tensor, float], oqs: List, _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.SingleGates import U2Gate

        _phi_info = phi.item() if isinstance(phi, Tensor) else phi
        _lam_info = lam.item() if isinstance(lam, Tensor) else lam
        _headline = f"U2{oqs}|(P{_phi_info:3f})-(L{_lam_info:3f})".replace('.', ';')

        self._add_module(U2Gate(phi, lam, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def u3(self, theta: Union[Tensor, float], phi: Union[Tensor, float], lam: Union[Tensor, float],
           oqs: List, _ideal: Optional[bool] = None):
        if isinstance(oqs, int):
            oqs = [oqs]
        from .QuantumGates.SingleGates import U3Gate

        _theta_info = theta.item() if isinstance(theta, Tensor) else theta
        _phi_info = phi.item() if isinstance(phi, Tensor) else phi
        _lam_info = lam.item() if isinstance(lam, Tensor) else lam
        _headline = f"U3{oqs}|(T{_theta_info:3f})-(P{_phi_info:3f})-(L{_lam_info:3f})".replace('.', ';')

        self._add_module(U3Gate(theta, phi, lam, _ideal, dtype=self.dtype, device=self.device), oqs, _headline)

    def truncate(self):
        """
        Add a truncation layer to the circuit.
        """
        self._oqs_list.append(['None'])
        self.layers.add_module(f'Truncation-{self._sequenceT}', None)
        self._sequenceT += 1

    def barrier(self):
        """
        Add a barrier to the circuit.
        """
        self._oqs_list.append(['None'])
        self.layers.add_module(f'Barrier-{self._sequenceT}', None)
        self._sequenceT += 1

    def reset0(self, oqs: Union[List, int]):
        if isinstance(oqs, int):
            oqs = [oqs]

        from .QuantumGates.SingleGates import Reset0
        _headline = f"Reset;0{oqs}|None"

        self._add_module(Reset0(dtype=self.dtype, device=self.device), oqs, _headline)

    def reset1(self, oqs: Union[List, int]):
        if isinstance(oqs, int):
            oqs = [oqs]

        from .QuantumGates.SingleGates import Reset1
        _headline = f"Reset;1{oqs}|None"

        self._add_module(Reset1(dtype=self.dtype, device=self.device), oqs, _headline)

    def measure(self, oqs: Union[List, int], orientations: Optional[Union[List, int]] = None):
        if orientations is None:
            orientations = [2] * len(oqs)
        if isinstance(oqs, int):
            oqs = [oqs]
        if isinstance(orientations, int):
            orientations = [orientations]

        from .QuantumGates.SingleGates import MeasureX, MeasureY, MeasureZ
        _measure_map = {0: MeasureX, 1: MeasureY, 2: MeasureZ}

        for oq, ori in zip(oqs, orientations):
            if ori not in _measure_map:
                raise ValueError("Orientation beyond the settings.")

            _headline = f"Measure;{oq}|Orientation;{ori}"
            self._add_module(_measure_map[ori](dtype=self.dtype, device=self.device), [oq], _headline)
