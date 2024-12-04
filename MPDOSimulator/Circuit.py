"""
Author: weiguo_ma
Time: 11.23.2024
Contact: weiguo.m@iphy.ac.cn
"""
from multiprocessing.managers import Value
from typing import Optional, Dict, Union, List, Any, Tuple

import tensornetwork as tn
import torch as tc
from tqdm import tqdm

from .AbstractCircuit import QuantumCircuit
from .NoiseChannel import NoiseChannel
from .QuantumGates.AbstractGate import QuantumGate
from .TNNOptimizer import bondTruncate, svdKappa_left2right, checkConnectivity
from .Tools import EdgeName2AxisName, count_item
from .dmOperations import reduce_dmNodes


class TensorCircuit(QuantumCircuit):
    def __init__(self, qn: int,
                 ideal: bool = True,
                 noiseType: str = 'no',
                 chiFileDict: Optional[Dict[str, Dict[str, Any]]] = None,
                 chi: Optional[int] = None, kappa: Optional[int] = None,
                 tnn_optimize: bool = True,
                 chip: Optional[str] = None,
                 dtype: Optional = tc.complex64,
                 device: Optional[Union[str, int]] = None):
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
        self.realNoise = True if (noiseType == 'realNoise' and not ideal) else False

        super(TensorCircuit, self).__init__(
            self.realNoise, noiseFiles=chiFileDict,
            chi=chi, kappa=kappa, tnn_optimize=tnn_optimize, dtype=dtype, device=device
        )

        self.qnumber = qn

        # Noisy Circuit Setting
        self.ideal = ideal
        self.unified, self.idealNoise = False, False

        if not self.ideal:
            self.Noise = NoiseChannel(chip=chip, dtype=self.dtype, device=self.device)
            if noiseType == 'unified':
                self.unified = True
            elif noiseType == 'realNoise':
                self.realNoise = True
            elif noiseType == 'idealNoise':
                self.idealNoise = True
            else:
                raise TypeError(f'Noise type "{noiseType}" is not supported yet.')

    @staticmethod
    def _assignEdges(contracted_node: tn.AbstractNode, minIdx: int):
        _lEdges, _rEdges = [], []
        for _axis in contracted_node.axis_names:
            if f'_{minIdx}' in _axis:
                _lEdges.append(contracted_node[_axis])
            else:
                _rEdges.append(contracted_node[_axis])
        return _lEdges, _rEdges

    def _add_gate(self, _qubits: List[tn.AbstractNode], _layer_num: int, _oqs: List[int]):
        r"""
        Add quantum Gate to tensornetwork

        Args:
            _oqs: operating qubits.

        Returns:
            Tensornetwork after adding gate.
        """
        _qNodes = _qubits
        _qubits = [_qubit[f'physics_{_idx}'] for _idx, _qubit in enumerate(_qubits)]
        gate = self.layers[_layer_num]

        if gate.name == 'MeasureZ':
            return None  # Lazy Code
        if not gate:  # Stopping condition
            return None

        _maxIdx, _minIdx = max(_oqs), min(_oqs)

        if not isinstance(_qubits, List):
            raise TypeError('Qubit must be a list of nodes.')
        if not isinstance(gate, QuantumGate):
            raise TypeError(f'Gate must be a TensorGate, current type is {type(gate)}.')
        if not isinstance(_oqs, List):
            raise TypeError('Operating qubits must be a list.')
        if _maxIdx >= self.qnumber:
            raise ValueError(f'Qubit index out of range, max index is Q{_maxIdx}.')

        single = gate.single

        if not single:  # Two-qubit gate
            """
                           |p0  |p1                         | w              | p
                        ___|____|___                    ____|____        ____|____
                        |          |                    |       |        |       |
                        |    DG    |---- I        l ----| Qubit |--------| Qubit |---- r
                        |__________|                    |_______|    k   |_______|
                           |    |                           |                |
                           |i0  |i1                         | m              | n
            """
            if len(_oqs) != 2 and _oqs[0] == _oqs[1]:
                raise ValueError('Invalid operating qubits for a two-qubit gate.')

            _qNoise = True if f'I_{_maxIdx}' in _qNodes[_maxIdx].axis_names else False
            _gNoise = (self.idealNoise and not gate.ideal) or self.realNoise

            # Adding depolarization noise channel to Two-qubit gate
            _gateNode = (
                tn.Node(tc.einsum('ijklp, klmn -> ijmnp', self.Noise.dpCTensor2,
                                  gate.tensor), name=gate.name,
                        axis_names=[f'physics_{_oqs[0]}', f'physics_{_oqs[1]}',
                                    f'inner_{_oqs[0]}', f'inner_{_oqs[1]}', f'I_{_maxIdx}{'_N' * _qNoise}'])
                if _gNoise and not self.realNoise else
                tn.Node(
                    gate.tensor, name=gate.name,
                    axis_names=[f'physics_{_oqs[0]}', f'physics_{_oqs[1]}', f'inner_{_oqs[0]}', f'inner_{_oqs[1]}'] +
                               [f'I_{_maxIdx}{'_N' * _qNoise}'] * self.realNoise
                )
            )

            _gateNode[f'inner_{_minIdx}'] ^ _qubits[_minIdx], _gateNode[f'inner_{_maxIdx}'] ^ _qubits[_maxIdx]

            _contracted_node = tn.contractors.auto(
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
            _qNodes[_minIdx], _qNodes[_maxIdx], _ = tn.split_node(
                _contracted_node, left_edges=_lEdges, right_edges=_rEdges,
                left_name=_qNodes[_minIdx].name, right_name=_qNodes[_maxIdx].name,
                edge_name=f'bond_{_minIdx}_{_maxIdx}'
            )
            EdgeName2AxisName([_qNodes[_minIdx], _qNodes[_maxIdx]])

        else:
            """
                            | m                             | i
                        ____|___                        ____|____
                        |      |                        |       |   
                        |  SG  |---- n            l ----| Qubit |---- r
                        |______|                        |_______|
                            |                               |
                            | i                             | j
            """
            _qNoiseList = [f'I_{_idx}' in _qNodes[_idx].axis_names for _idx in _oqs]
            _gNoiseList = [(self.idealNoise or self.unified) and not gate.ideal for _ in _oqs]

            gate_list = [
                tn.Node(tc.reshape(
                    tc.einsum('nlm, ljk, ji -> nimk', self.Noise.dpCTensor, self.Noise.apdeCTensor, gate.tensor),
                    (2, 2, -1)), name=gate.name,
                    axis_names=[f'physics_{_idx}', f'inner_{_idx}', f'I_{_idx}{'_N' * _qNoiseList[_j]}'])
                if _gNoiseList[_j] else
                tn.Node(gate.tensor, name=gate.name, axis_names=[f'physics_{_idx}', f'inner_{_idx}'])
                for _j, _idx in enumerate(_oqs)
            ]

            for _i, _bit in enumerate(_oqs):
                _gate = gate_list[_i]
                _connected_edge = _gate[f'inner_{_bit}'] ^ _qubits[_bit]

                _qNodes[_bit] = tn.contract(_connected_edge, name=_qNodes[_bit].name)
                EdgeName2AxisName([_qNodes[_bit]])

                if _gNoiseList[_i] and _qNoiseList[_i]:
                    tn.flatten_edges(
                        edges=[_qNodes[_bit][f'I_{_bit}'], _qNodes[_bit][f'I_{_bit}_N']], new_edge_name=f'I_{_bit}'
                    )
                    EdgeName2AxisName([_qNodes[_bit]])

    def _create_dmNodes(self, reduced_index: Optional[List] = None):
        _state, _qubits_conj = tn.replicate_nodes(self._stateNodes), tn.replicate_nodes(self._stateNodes)
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

        _state, _qubits_conj = self._create_dmNodes(reduced_index)
        self._dmNodes = _state + _qubits_conj
        return self._dmNodes

    def cal_dm(self, reduced_index: Optional[List] = None):
        if reduced_index is None:
            reduced_index = []

        if self._dmNodes is None:
            _state, _qubits_conj = self._create_dmNodes()
            self._dmNodes = tn.replicate_nodes(_state + _qubits_conj)
        else:
            _nodes = tn.replicate_nodes(self._dmNodes)
            _state, _qubits_conj = _nodes[:self.qnumber], _nodes[self.qnumber:]

        _dm = self._contract_dm(_state, _qubits_conj, reduced_index)
        self._dm = _dm
        return _dm

    def _conditional_sample(self,
                            _history: List[int],
                            _dmNodes: List[tn.AbstractNode],
                            _conj_dmNodes: List[tn.AbstractNode],
                            _idx: int, _ori_list: List[int], _orientations: List[int]) -> int:
        _ignored_reduced = _ori_list[_idx + 1:]

        _contract_nodes = []
        with tn.NodeCollection(_contract_nodes):
            _proj_s = [
                (
                    tn.replicate_nodes([self._projectors[_orientations[_hisIdx]][0]])[0],
                    tn.replicate_nodes([self._projectors[_orientations[_hisIdx]][0]])[0]
                )
                if _his == 0 else (
                    tn.replicate_nodes([self._projectors[_orientations[_hisIdx]][1]])[0],
                    tn.replicate_nodes([self._projectors[_orientations[_hisIdx]][1]])[0]
                )
                for _hisIdx, _his in enumerate(_history)
            ]

            _nodes_intermediate = tn.replicate_nodes([*_dmNodes, *_conj_dmNodes])
            _dmNodes_intermediate, _conj_dmNodes_intermediate = (
                _nodes_intermediate[:self.qnumber], _nodes_intermediate[self.qnumber:]
            )
            reduce_dmNodes(_dmNodes_intermediate, _conj_dmNodes_intermediate, reduced_index=_ignored_reduced)

            for _j, (_proj, _con_proj) in enumerate(_proj_s):
                _proj['proj'] ^ _dmNodes_intermediate[_ori_list[_j]][f'physics_{_ori_list[_j]}']
                _con_proj['proj'] ^ _conj_dmNodes_intermediate[_ori_list[_j]][f'con_physics_{_ori_list[_j]}']

        _probs = tc.diag(
            tn.contractors.auto(
                _contract_nodes, output_edge_order=[
                    _dmNodes_intermediate[_ori_list[_idx]][f'physics_{_ori_list[_idx]}'],
                    _conj_dmNodes_intermediate[_ori_list[_idx]][f'con_physics_{_ori_list[_idx]}']
                ]
            ).tensor
        )
        try:
            return tc.multinomial(_probs.real, num_samples=1).item()
        except RuntimeError:
            raise RuntimeError("State is illegal, all probs. are 0.")

    def fakeSample(self,
                   shots: Optional[int] = None,
                   orientation: Optional[List[int]] = None,
                   reduced: Optional[List[int]] = None, sample_string: bool = True) -> Tuple[List[List[int]], Dict]:
        shots = 1024 if shots is None else shots

        reduced = [] if reduced is None else reduced
        _ori_list = [num for num in range(self.qnumber) if num not in reduced]
        _sampleLength = len(_ori_list)

        orientation = [2] * _sampleLength if orientation is None else orientation
        if len(orientation) != _sampleLength:
            raise ValueError(
                "Length of orientation must be equal to sample length. "
                "You are probably sampling from a reduced qubit, or providing insufficient orientations."
            )

        self._load_projectors()  # Load projectors in memory.

        _dmNodes, _conj_dmNodes = self._create_dmNodes()
        reduce_dmNodes(_dmNodes, _conj_dmNodes, reduced_index=reduced)

        _bitStrings = [[] for _ in range(shots)]
        for _i in tqdm(
                range(shots),
                desc=f"Processing Shots for scheme - {''.join([self._projectors_string[num] for num in orientation])}"
        ):
            _choices = [-1] * _sampleLength
            for _j in range(_sampleLength):     # _j: index in un-reduced dmNodes.
                _choices[_j] = self._conditional_sample(
                    _choices[:_j], _dmNodes, _conj_dmNodes,
                    _idx=_j, _ori_list=_ori_list, _orientations=orientation
                )
            _bitStrings[_i] = _choices

        if sample_string:
            _bitStrings = [''.join(map(str, _stringList)) for _stringList in _bitStrings]

        self._samples = _bitStrings
        self._counts = count_item(_bitStrings)

        return self._samples, self._counts

    def randomSample(self, measurement_schemes: List[List[int]], shots_per_scheme: int = 1024) -> List[List[List[int]]]:
        """
        Args:
            measurement_schemes: __len__ == M.
            shots_per_scheme: K

        Returns:
            Measurement results.
        """
        _measurement_outcomes = [
            self.fakeSample(shots=shots_per_scheme, orientation=_scheme, sample_string=False)[0]
            for _scheme in measurement_schemes
        ]

        return _measurement_outcomes

    def evolve(self, state: List[tn.AbstractNode]):
        self._initState = tn.replicate_nodes(state)
        [state.set_tensor(state.tensor.to(dtype=self.dtype, device=self.device)) for state in self._initState]

        for _i, layer in enumerate(self.layers):
            self._add_gate(state, _i, _oqs=self._oqs_list[_i])
            self.Truncate = True if layer is None else False
            #
            if self.Truncate and self.tnn_optimize:
                if checkConnectivity(state) and self.chi is not None:
                    bondTruncate(state, self.chi)
                if not self.ideal and self.kappa is not None:
                    svdKappa_left2right(state, kappa=self.kappa)

            self.Truncate = False

        # LastLayer noise-truncation
        if self.tnn_optimize and not self.ideal and self.kappa is not None:
            svdKappa_left2right(state, kappa=self.kappa)

        self._stateNodes = state  # Same memory id()

    # Hidden interface
    def forward(self, state: List[tn.AbstractNode]):
        """
        Forward propagation of tensornetwork.
        """
        self.evolve(state)
