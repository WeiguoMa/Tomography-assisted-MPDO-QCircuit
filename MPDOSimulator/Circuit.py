"""
Author: weiguo_ma
Time: 11.23.2024
Contact: weiguo.m@iphy.ac.cn
"""
from typing import Optional, Dict, Union, List, Any, Tuple

import tensornetwork as tn
import torch as tc

from .AbstractCircuit import QuantumCircuit
from .NoiseChannel import NoiseChannel
from .QuantumGates.AbstractGate import QuantumGate
from .TNNOptimizer import bondTruncate, svdKappa_left2right, checkConnectivity
from .Tools import select_device, EdgeName2AxisName


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
        self.device = select_device(device)
        self.dtype = dtype

        super(TensorCircuit, self).__init__(
            self.realNoise, noiseFiles=chiFileDict, dtype=self.dtype, device=self.device
        )

        # About program
        self.fVR = None
        self.qnumber = qn
        self.state = None

        # Noisy Circuit Setting
        self.ideal = ideal
        self.unified, self.idealNoise = False, False

        if not self.ideal:
            self.Noise = NoiseChannel(chip=chip, device=self.device)
            if noiseType == 'unified':
                self.unified = True
            elif noiseType == 'realNoise':
                self.realNoise = True
            elif noiseType == 'idealNoise':
                self.idealNoise = True
            else:
                raise TypeError(f'Noise type "{noiseType}" is not supported yet.')

        # Density Matrix
        self.DM = None
        self.DMNode = None

        # TNN Truncation
        self.chi = chi
        self.kappa = kappa
        self.tnn_optimize = tnn_optimize

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

    def _calculate_DM(self, state_vector: bool = False,
                      reduced_index: Optional[List[Union[List[int], int]]] = None) -> tc.Tensor:
        """
        Calculate the density matrix of the state.

        Args:
            state_vector: if True, the state is a state vector, otherwise, the state is a density matrix.
            reduced_index: the state[index] to be reduced, which means the physics_con-physics of sub-density matrix
                                will be connected and contracted.

        Returns:
            _dm: the density matrix node;
            _dm_tensor: the density matrix tensor.

        Additional information:
            A mixed state(noisy situation) is generally cannot be efficiently represented by a state vector but DM.

        Raises:
            ValueError: if the state is chosen to be a vector but is noisy;
            ValueError: Reduced index should be empty as [] or None.
        """
        if not reduced_index:
            reduced_index = None

        if reduced_index and max(reduced_index) >= self.qnumber:
            raise ValueError('Reduced index should not be larger than the qubit number.')

        if not state_vector:
            _qubits_conj = tn.replicate_nodes(self.state)
            for _qubit in _qubits_conj:
                _qubit.set_tensor(_qubit.tensor.conj())

            # Differential name the conjugate qubits' edges name to permute the order of the indices
            for _i, _qubit_conj in enumerate(_qubits_conj):
                _qubit_conj.set_name(f'con_{_qubit_conj.name}')
                for _ii, _edge in enumerate(_qubit_conj.edges):
                    if 'physics' in _edge.name:
                        _edge.set_name(f'con_{_qubit_conj[_ii].name}')
                        _qubit_conj.axis_names[_ii] = f'con_{_qubit_conj.axis_names[_ii]}'

            for i in range(len(self.state)):
                if not self.ideal and f'I_{i}' in self.state[i].axis_names:
                    tn.connect(self.state[i][f'I_{i}'], _qubits_conj[i][f'I_{i}'])

            # Reduced density matrix
            if reduced_index:
                reduced_index = [reduced_index] if isinstance(reduced_index, int) else reduced_index
                if not isinstance(reduced_index, list):
                    raise TypeError('reduced_index should be int or list[int]')
                for _idx in reduced_index:
                    tn.connect(self.state[_idx][f'physics_{_idx}'], _qubits_conj[_idx][f'con_physics_{_idx}'])
            else:
                reduced_index = []

            _numList = [_i for _i in range(self.qnumber) if _i not in reduced_index]
            _output_edge_order = (
                    [self.state[i][f'physics_{i}'] for i in _numList] +
                    [_qubits_conj[i][f'con_physics_{i}'] for i in _numList]
            )
            _dm = tn.contractors.auto(self.state + _qubits_conj, output_edge_order=_output_edge_order)

            _reshape_size = self.qnumber - len(reduced_index)
            self.DM, self.DMNode = _dm.tensor.reshape((2 ** _reshape_size, 2 ** _reshape_size)), self.DMNode
            return self.DM

        else:
            if not self.fVR and not self.ideal:
                raise ValueError('Noisy circuit cannot be represented by state vector efficiently.')
            if reduced_index is not None:
                raise ValueError('State vector cannot efficiently represents the reduced density matrix.')

            _outOrder = [self.state[i][f'physics_{i}'] for i in list(range(self.qnumber))]
            _vector = tn.contractors.auto(self.state, output_edge_order=_outOrder)

            self.DM = _vector.tensor.reshape((2 ** self.qnumber, 1)) if not self.fVR else _vector.tensor
            return self.DM

    def forward(self,
                state: List[tn.AbstractNode],
                require_nodes: bool = False,
                density_matrix: bool = True,
                state_vector: bool = False,
                reduced_index: Optional[List] = None,
                forceVectorRequire: bool = False) -> Union[tc.Tensor, Dict, Tuple]:
        """
        Forward propagation of tensornetwork.

        Returns:
            self.state: tensornetwork after forward propagation.
        """
        self.state = state
        self.qnumber, self.fVR = len(state), forceVectorRequire

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

        _nodes = tn.replicate_nodes(self.state) if require_nodes else None
        _dm = self._calculate_DM(state_vector=state_vector, reduced_index=reduced_index) if density_matrix else None

        return (_nodes, _dm) if require_nodes else _dm
