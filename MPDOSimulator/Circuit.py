"""
Author: weiguo_ma
Time: 11.23.2024
Contact: weiguo.m@iphy.ac.cn
"""
from typing import Optional, Dict, Union, List, Any, Tuple

import tensornetwork as tn
import torch as tc
from tensornetwork import AbstractNode
from torch.linalg import LinAlgError
from tqdm import tqdm, trange

from .AbstractCircuit import QuantumCircuit
from .NoiseChannel import NoiseChannel
from .QuantumGates.AbstractGate import QuantumGate
from .QuantumGates.SingleGates import MeasureX, MeasureY
from .TNNOptimizer import bondTruncate, svdKappa_left2right, checkConnectivity
from .Tools import EdgeName2AxisName, count_item
from .dmOperations import reduce_dmNodes

GLOBAL_MINIMUM = 2.718281828459045 * 1e-8
MAX_ATTEMPTS = 4


class TensorCircuit(QuantumCircuit):
    def __init__(self, qn: int, ideal: bool = True, noiseType: str = 'no',
                 chiFileDict: Optional[Dict[str, Dict[str, Any]]] = None,
                 chi: Optional[int] = None, kappa: Optional[int] = None,
                 max_truncation_err: Optional[float] = None,
                 tnn_optimize: bool = True, chip: Optional[str] = None,
                 dtype: Optional = tc.complex64, device: Optional[Union[str, int]] = None):
        """
        Args:
            ideal: Whether the circuit is ideal.  --> Whether the one-qubit gate is ideal or not.
            noiseType: The type of the noise channel.   --> Which kind of noise channel is added to the two-qubit gate.
            chiFileDict: The filename of the chi matrix.
            chi: The chi parameter of the TNN.
            kappa: The kappa parameter of the TNN.
            tnn_optimize: Whether the TNN is optimized.
            chip: The name of the chip.
            device: The device of the torch;
        """
        super(TensorCircuit, self).__init__(
            noiseFiles=chiFileDict,
            chi=chi, kappa=kappa, max_truncation_err=max_truncation_err, dtype=dtype, device=device
        )

        self.qnumber = qn
        self.ideal = ideal
        self.noiseType = noiseType.lower()
        self.Noise = None

        if not ideal:
            if self.noiseType not in ['unified', 'realnoise', 'idealnoise']:
                raise ValueError(f'Unsupported noise type: {self.noiseType}')
            self.Noise = NoiseChannel(chip=chip, dtype=dtype, device=device)
            self.unified = self.noiseType == 'unified'
            self.realNoise = self.noiseType == 'realnoise'
            self.idealNoise = self.noiseType == 'idealnoise'

            if self.realNoise:
                self._load_exp_tensors()

    @staticmethod
    def _assignEdges(contracted_node: tn.AbstractNode, minIdx: int):
        _lEdges, _rEdges = [], []
        for _axis in contracted_node.axis_names:
            if f'_{minIdx}' in _axis:
                _lEdges.append(contracted_node[_axis])
            else:
                _rEdges.append(contracted_node[_axis])
        return _lEdges, _rEdges

    def _apply_two_qubits_gate(self, _qNodes: List[tn.AbstractNode],
                               _qubits: List[tn.Edge], gate: QuantumGate, _oqs: List[int]):
        """
            Apply a two-qubit gate to the circuit.
        """
        _maxIdx, _minIdx = max(_oqs), min(_oqs)
        if len(_oqs) != 2 and _oqs[0] == _oqs[1]:
            raise ValueError('Invalid operating qubits for a two-qubit gate.')

        _qNoise = True if f'I_{_maxIdx}' in _qNodes[_maxIdx].axis_names else False
        _gNoise = (self.idealNoise and not gate.ideal) or self.realNoise

        # Adding depolarization noise channel to Two-qubit gate
        _gTensor = gate.tensor
        _gAxis = [f'physics_{_oqs[0]}', f'physics_{_oqs[1]}', f'inner_{_oqs[0]}', f'inner_{_oqs[1]}'] + [
            f'I_{_maxIdx}{'_N' * _qNoise}'] * self.realNoise

        if _gNoise and not self.realNoise:
            if not gate.variational:
                _gTensor = self.noiseTensorDict.setdefault(
                    gate.name,
                    tc.einsum('ijklp, klmn -> ijmnp', self.Noise.dpCTensor2, _gTensor)
                )
                _gAxis.append(f'I_{_maxIdx}{'_N' * _qNoise}')
            else:
                _gTensor = tc.einsum('ijklp, klmn -> ijmnp', self.Noise.dpCTensor2, _gTensor)

        _gateNode = tn.Node(_gTensor, name=gate.name, axis_names=_gAxis)
        _gateNode[f'inner_{_minIdx}'] ^ _qubits[_minIdx], _gateNode[f'inner_{_maxIdx}'] ^ _qubits[_maxIdx]

        _contracted_node = tn.contractors.optimal(
            [_qNodes[_minIdx], _qNodes[_maxIdx], _gateNode], ignore_edge_order=True
        )
        EdgeName2AxisName([_contracted_node])

        if _gNoise and _qNoise:
            tn.flatten_edges(
                edges=[_contracted_node[f'I_{_maxIdx}'], _contracted_node[f'I_{_maxIdx}_N']],
                new_edge_name=f'I_{_maxIdx}'
            )
            EdgeName2AxisName([_contracted_node])

        _lEdges, _rEdges = self._assignEdges(_contracted_node, _minIdx)

        for attempt in range(MAX_ATTEMPTS):
            try:
                _qNodes[_minIdx], _qNodes[_maxIdx], _ = tn.split_node(
                    _contracted_node, left_edges=_lEdges, right_edges=_rEdges,
                    left_name=_qNodes[_minIdx].name, right_name=_qNodes[_maxIdx].name,
                    edge_name=f'bond_{_minIdx}_{_maxIdx}', max_truncation_err=GLOBAL_MINIMUM
                )
                EdgeName2AxisName([_qNodes[_minIdx], _qNodes[_maxIdx]])
                break
            except LinAlgError:
                _tensor_norm = tc.norm(_contracted_node.tensor).item()
                _scale_factor = GLOBAL_MINIMUM * 1e-2
                _perturbation_magnitude = tc.finfo(tc.float64).eps * max(_tensor_norm, 1.0) * _scale_factor
                _perturbation = _perturbation_magnitude * (
                        tc.randn_like(_contracted_node.tensor) + 1j * tc.randn_like(_contracted_node.tensor)
                )
                _contracted_node.set_tensor(_contracted_node.tensor + _perturbation)
        else:
            raise LinAlgError(f'SVD converges failed. Multiple times ({MAX_ATTEMPTS}) of perturbation are tried.')

    def _apply_single_qubit_gate(self, _qNodes: List[tn.AbstractNode],
                                 _qubits: List[tn.Edge], gate: QuantumGate, _oqs: Union[int, List[int]]):
        _qNoiseList = [f'I_{_idx}' in _qNodes[_idx].axis_names for _idx in _oqs]
        _gNoise = (self.idealNoise or self.unified) and not gate.ideal

        _gTensor = gate.tensor
        if _gNoise:
            if not gate.variational:
                _gTensor = self.noiseTensorDict.setdefault(
                    gate.name,
                    tc.einsum(
                        'nlm, ljk, ji -> nimk', self.Noise.decayTensor, self.Noise.dephasingTensor, _gTensor
                    ).reshape((2, 2, -1))
                )
            else:
                _gTensor = tc.einsum(
                    'nlm, ljk, ji -> nimk', self.Noise.decayTensor, self.Noise.dephasingTensor, _gTensor
                ).reshape((2, 2, -1))

        gate_list = [
            tn.Node(
                _gTensor, name=gate.name,
                axis_names=[f'physics_{_idx}', f'inner_{_idx}', f'I_{_idx}{'_N' * _qNoiseList[_j]}']
            )
            if _gNoise else
            tn.Node(_gTensor, name=gate.name, axis_names=[f'physics_{_idx}', f'inner_{_idx}'])
            for _j, _idx in enumerate(_oqs)
        ]

        for _i, _bit in enumerate(_oqs):
            _gate = gate_list[_i]
            _connected_edge = _gate[f'inner_{_bit}'] ^ _qubits[_bit]

            _qNodes[_bit] = tn.contract(_connected_edge, name=_qNodes[_bit].name)
            EdgeName2AxisName([_qNodes[_bit]])

            if _gNoise and _qNoiseList[_i]:
                tn.flatten_edges(
                    edges=[_qNodes[_bit][f'I_{_bit}'], _qNodes[_bit][f'I_{_bit}_N']], new_edge_name=f'I_{_bit}'
                )
                EdgeName2AxisName([_qNodes[_bit]])

    def _add_gate(self, _qubits: List[tn.AbstractNode], _layer_num: int, _oqs: List[int], _gate: Optional = None):
        """
        Add quantum Gate.
        """
        if not isinstance(_qubits, List):
            raise TypeError('Qubit must be a list of nodes.')
        if not isinstance(_oqs, List):
            raise TypeError('Operating qubits must be a list.')
        if max(_oqs) is None:
            return None
        if max(_oqs) >= self.qnumber:
            raise ValueError(f'Qubit index out of range, max index is Q{max(_oqs)}.')

        _qNodes = _qubits
        _qubits = [_qubit[f'physics_{_idx}'] for _idx, _qubit in enumerate(_qubits)]

        gate = _gate or self.layers[_layer_num]
        if not gate or gate.name == 'MeasureZ':  # Stopping condition
            return None
        if not isinstance(gate, QuantumGate):
            raise TypeError(f'Gate must be a QuantumGate, current type is {type(gate)}.')

        single = gate.single
        if not single:
            """
                           |p0  |p1                                 | p_q            | p_{q+1}
                        ___|____|___                            ____|____        ____|____
                        |          |                            |       |        |       |
                        |    DG    |---- I   Bond_{q-1}_{q} ----| Qubit |--------| Qubit |---- Bond_{q+1}_{q+2}
                        |__________|                            |_______|    B   |_______|
                           |    |                                   |                |
                           |i0  |i1                                 | i_q            | i_{q+1}
            """
            self._apply_two_qubits_gate(_qNodes, _qubits, gate, _oqs)
        else:
            """
                            | p                                  | p_q
                        ____|___                             ____|____
                        |      |                             |       |   
                        |  SG  |---- I    Bond_{q-1}_{q} ----| Qubit |---- Bond_{q}_{q+1}
                        |______|                             |_______|
                            |                                    |
                            | i                                  | i_q
            """
            self._apply_single_qubit_gate(_qNodes, _qubits, gate, _oqs)

    def _create_dmNodes(self, _stateNodes: Optional[List[AbstractNode]] = None, reduced_index: Optional[List] = None):
        _state = tn.replicate_nodes(_stateNodes) if _stateNodes is not None else tn.replicate_nodes(self._stateNodes)
        _qubits_conj = tn.replicate_nodes(_stateNodes) if _stateNodes is not None else tn.replicate_nodes(
            self._stateNodes)
        for _qubit in _qubits_conj:  # make conj(), which is much faster than conjugate=True in replicate_nodes()
            _qubit.set_tensor(_qubit.tensor.conj())

        for _i, _qubit_conj in enumerate(_qubits_conj):
            _qubit_conj.set_name(f'con_{_qubit_conj.name}')
            for _ii, _edge in enumerate(_qubit_conj.edges):
                if 'physics' in _edge.name:
                    _edge.set_name(f'con_{_qubit_conj[_ii].name}')
                    _qubit_conj.axis_names[_ii] = f'con_{_qubit_conj.axis_names[_ii]}'

        for i in range(self.qnumber):
            if not self.ideal and f'I_{i}' in _state[i].axis_names:
                _state[i][f'I_{i}'] ^ _qubits_conj[i][f'I_{i}']

        reduce_dmNodes(_state, _qubits_conj, reduced_index)
        return _state, _qubits_conj

    def _contract_dm(self,
                     _state: List[tn.AbstractNode],
                     _qubits_conj: List[tn.AbstractNode],
                     reduced_index: Optional[List] = None):
        _reshape_size = 2 ** (self.qnumber - len(reduced_index))
        _numList = [_i for _i in range(self.qnumber) if _i not in reduced_index]
        _output_edge_order = (
                [_state[i][f'physics_{i}'] for i in _numList] +
                [_qubits_conj[i][f'con_physics_{i}'] for i in _numList]
        )

        _dm = tn.contractors.auto(
            _state + _qubits_conj, output_edge_order=_output_edge_order
        ).tensor.reshape((_reshape_size, _reshape_size))

        return _dm

    def cal_vector(self):
        if not self.ideal:
            raise ValueError('Noisy circuit cannot be represented by state vector efficiently.')

        _state = tn.replicate_nodes(self._stateNodes)
        _outOrder = [_state[i][f'physics_{i}'] for i in list(range(self.qnumber))]
        _vector = tn.contractors.auto(_state, output_edge_order=_outOrder).tensor.reshape((2 ** self.qnumber, 1))
        self._vector = _vector
        return _vector

    def cal_dmNodes(self, reduced_index: Optional[List] = None):
        if reduced_index is None:
            reduced_index = []

        _state, _qubits_conj = self._create_dmNodes(reduced_index=reduced_index)
        self._dmNodes = _state + _qubits_conj
        return self._dmNodes

    def cal_dm(self, reduced_index: Optional[List] = None, _replicate_require: bool = False):
        if reduced_index is None:
            reduced_index = []

        if self._dmNodes is None or _replicate_require:
            _state, _qubits_conj = self._create_dmNodes(reduced_index=reduced_index)
            self._dmNodes = tn.replicate_nodes(_state + _qubits_conj)
        else:
            _nodes = tn.replicate_nodes(self._dmNodes)
            _state, _qubits_conj = _nodes[:self.qnumber], _nodes[self.qnumber:]

        _dm = self._contract_dm(_state, _qubits_conj, reduced_index)
        self._dm = _dm
        return _dm

    def _conditional_prb(self, _history: List[Union[int, bool]]) -> float:
        _ori_list: List[int] = self._indices4samples
        _idx = len(_history)
        _qLoc, _ignored_reduced = _ori_list[_idx], _ori_list[_idx + 1:]

        _contract_nodes = []
        with tn.NodeCollection(_contract_nodes):
            _proj_s = [
                tn.replicate_nodes(self._projectors[0]) if not _his else tn.replicate_nodes(self._projectors[1])
                for _hisIdx, _his in enumerate(_history)
            ]

            _dmNodes, _conj_dmNodes = self._nodes4samples
            _nodes_intermediate = tn.replicate_nodes([*_dmNodes, *_conj_dmNodes])
            _dmNodes_intermediate, _conj_dmNodes_intermediate = (
                _nodes_intermediate[:self.qnumber], _nodes_intermediate[self.qnumber:]
            )
            reduce_dmNodes(_dmNodes_intermediate, _conj_dmNodes_intermediate, reduced_index=_ignored_reduced)

            for _j, (_proj, _con_proj) in enumerate(_proj_s):
                _loc = _ori_list[_j]
                _proj['proj'] ^ _dmNodes_intermediate[_loc][f'physics_{_loc}']
                _con_proj['proj'] ^ _conj_dmNodes_intermediate[_loc][f'con_physics_{_loc}']

        _probs = tc.diag(
            tn.contractors.auto(_contract_nodes, output_edge_order=[
                _dmNodes_intermediate[_qLoc][f'physics_{_qLoc}'],
                _conj_dmNodes_intermediate[_qLoc][f'con_physics_{_qLoc}']
            ]).tensor
        )

        _probs = _probs.real + GLOBAL_MINIMUM
        if (_probs < 0).sum() > 0:
            raise RuntimeError(f"State is illegal, and your probability distribution is {_probs}.")

        return (_probs / _probs.sum())[-1].item()

    def _conditional_sample(self, shots: int, _sampleLength: int, _tqdm_disable: bool = False) -> List[List[int]]:
        _bitStrings = [[]] * shots
        for _i in trange(shots, desc=f"Sequential Sampling", disable=_tqdm_disable):
            _choices = [-1] * _sampleLength
            for _j in range(_sampleLength):
                _prob = self._conditional_prb(_choices[:_j])
                _choices[_j] = tc.multinomial(tc.tensor([1 - _prob, _prob]), num_samples=1).item()
            _bitStrings[_i] = _choices
        return _bitStrings

    def _conditional_batch_sample(
            self, shots: int, _sampleLength: int, _bool: bool = False, _tqdm_disable: bool = False
    ) -> List[List[int]]:
        if not _bool:
            _inner_dtype = tc.int
            _saving_value = 1
        else:
            _inner_dtype = tc.bool
            _saving_value = True

        _sequences = tc.zeros((shots, _sampleLength), dtype=_inner_dtype)

        _layers = [self.Group(history=[], start=0, length=shots)]
        for _qLoc in trange(_sampleLength, desc=f"Batch Sampling", disable=_tqdm_disable, smoothing=1):
            _new_layer = []
            for group in _layers:
                if group.length == 0:
                    continue

                p1 = self._conditional_prb(group.history)
                if p1 < GLOBAL_MINIMUM:
                    _k1, _k0 = 0, group.length
                elif p1 > 1 - GLOBAL_MINIMUM:
                    _k1, _k0 = group.length, 0
                else:
                    _k1 = int(tc.bernoulli(tc.full((group.length,), p1)).sum())
                    _k0 = group.length - _k1

                _sequences[group.start: group.start + _k1, _qLoc] = _saving_value

                if _qLoc < _sampleLength - 1:
                    if _k1 > 0:
                        one_history = group.history + [1]
                        _new_layer.append(
                            self.Group(history=one_history, start=group.start, length=_k1)
                        )
                    if _k0 > 0:
                        zero_history = group.history + [0]
                        _new_layer.append(
                            self.Group(history=zero_history, start=group.start + _k1, length=_k0)
                        )
            _layers = _new_layer

        return _sequences.tolist()

    def fakeSample(self,
                   shots: Optional[int] = None,
                   orientation: Optional[List[int]] = None,
                   reduced: Optional[List[int]] = None,
                   sample_string: bool = True, _tqdm_disable: bool = False,
                   _require_sequential_sample: bool = False,
                   _require_bool_result: bool = False, _require_counts: bool = True,
                   _stateNodes4Sample: Optional[List[tn.AbstractNode]] = None) -> Union[Tuple[List, Dict], List]:
        """
        Perform sampling with the specified number of shots and orientations.
        """
        shots = shots or 1024
        reduced = reduced or []
        _ori_list = [num for num in range(self.qnumber) if num not in reduced]
        _sampleLength = len(_ori_list)

        orientation = orientation or [2] * _sampleLength
        if len(orientation) != _sampleLength:
            raise ValueError(
                "Length of orientation must match the sample length. Check reduced or unmeasured qubits."
            )

        _nodes4samples = tn.replicate_nodes(self._stateNodes) if _stateNodes4Sample is None else tn.replicate_nodes(
            _stateNodes4Sample)
        for _ori_val, _measure_gate in [(0, MeasureX), (1, MeasureY)]:
            _indices = [i for i, value in enumerate(orientation) if value == _ori_val]
            if _indices:
                self._add_gate(
                    _nodes4samples, 0, _indices, _measure_gate(dtype=self.dtype, device=self.device)
                )

        _dmNodes, _conj_dmNodes = self._create_dmNodes(_nodes4samples)
        reduce_dmNodes(_dmNodes, _conj_dmNodes, reduced_index=reduced)
        self._nodes4samples, self._indices4samples = (_dmNodes, _conj_dmNodes), _ori_list

        if not _tqdm_disable:
            print(f'Sample Direction:'
                  f'\n (scheme-{''.join([self._projectors_string[num] for num in orientation])})')

        if _require_sequential_sample:
            _bitStrings = self._conditional_sample(
                shots=shots, _sampleLength=_sampleLength, _tqdm_disable=_tqdm_disable
            )
        else:
            _bitStrings = self._conditional_batch_sample(
                shots=shots, _sampleLength=_sampleLength, _tqdm_disable=_tqdm_disable, _bool=_require_bool_result
            )

        if sample_string and not _require_bool_result:
            _bitStrings = [''.join(map(str, _stringList)) for _stringList in _bitStrings]

        self._samples = _bitStrings

        if _require_counts:
            self._counts = count_item(_bitStrings)
            return self._samples, self._counts
        else:
            return self._samples

    def randomSample(self,
                     measurement_schemes: List[List[int]], shots_per_scheme: int = 1024,
                     reduced: Optional[List[int]] = None,
                     _tqdm_disable: bool = False, _require_sequential_sample: bool = False,
                     _require_bool_result: bool = False,
                     _stateNodes4Sample: Optional[List[tn.AbstractNode]] = None) -> List[List[List[Union[int, bool]]]]:
        """
        Perform random sampling for multiple measurement schemes.
        """
        _measurement_outcomes = [
            self.fakeSample(
                shots=shots_per_scheme, orientation=scheme,
                reduced=reduced, sample_string=False,
                _tqdm_disable=True, _require_sequential_sample=_require_sequential_sample,
                _require_bool_result=_require_bool_result, _require_counts=False,
                _stateNodes4Sample=_stateNodes4Sample
            )
            for scheme in tqdm(measurement_schemes, desc=f"Random Sampling", disable=_tqdm_disable)
        ]
        return _measurement_outcomes

    def evolve(self, state: List[tn.AbstractNode]):
        self._initState = tn.replicate_nodes(state)
        [state.set_tensor(state.tensor.to(dtype=self.dtype, device=self.device)) for state in self._initState]

        for _i, layer in enumerate(self.layers):
            layer_name = layer.name.lower()
            match layer_name:
                case name if "truncate" in name:
                    if checkConnectivity(state):
                        bondTruncate(state, max_singular_values=self.chi, max_truncation_err=self.max_truncation_err)
                        if not self.ideal:
                            svdKappa_left2right(state, max_singular_values=self.kappa,
                                                max_truncation_err=self.max_truncation_err)
                case name if "barrier" in name:
                    pass
                case _:
                    self._add_gate(state, _i, _oqs=self._oqs_list[_i])

        # LastLayer noise-truncation
        if not self.ideal and not 'truncate' in self.layers[-1].name:
            svdKappa_left2right(state, max_singular_values=self.kappa, max_truncation_err=self.max_truncation_err)

        self._stateNodes = state  # Same memory id()

    # Hidden interface
    def forward(self, state: List[tn.AbstractNode]):
        """
        Forward propagation of tensornetwork.
        """
        self.evolve(state)
