"""
Author: weiguo_ma
Time: 04.26.2023
Contact: weiguo.m@iphy.ac.cn
"""
from copy import deepcopy
from typing import Optional

import numpy as np
import tensornetwork as tn
import torch as tc
from torch import nn

from Library.ADGate import TensorGate
from Library.ADNGate import NoisyTensorGate
from Library.AbstractGate import AbstractGate
from Library.NoiseChannel import NoiseChannel
from Library.TNNOptimizer import svd_right2left, qr_left2right, checkConnectivity
from Library.realNoise import czExp_channel
from Library.tools import select_device, generate_random_string_without_duplicate, move_index, find_duplicate, \
	EdgeName2AxisName, is_nested


class TensorCircuit(nn.Module):
	def __init__(self, qn: int, ideal: bool = True, noiseType: str = 'no', chiFilename: str = None,
	             crossTalk: bool = False,
	             chi: Optional[int] = None, kappa: Optional[int] = None, tnn_optimize: bool = True,
	             chip: Optional[str] = None, device: str or int = 'cpu'):
		"""
		Args:
			ideal: Whether the circuit is ideal.  --> Whether the one-qubit gate is ideal or not.
			noiseType: The type of the noise channel.   --> Which kind of noise channel is added to the two-qubit gate.
			chiFilename: The filename of the chi matrix.
			chi: The chi parameter of the TNN.
			kappa: The kappa parameter of the TNN.
			tnn_optimize: Whether the TNN is optimized.
			chip: The name of the chip.
			device: The device of the torch.
		"""
		super(TensorCircuit, self).__init__()
		self.fVR = None
		self.i = 0
		self.qnumber = qn
		self.state = None
		self.initState = None

		# Noisy Circuit Setting
		self.ideal = ideal
		self.realNoise = False
		self.idealNoise = False
		self.unified = None
		self.crossTalk = crossTalk
		if self.ideal is False:
			self.Noise = NoiseChannel(chip=chip)
			if noiseType == 'unified':
				self.unified = True
			elif noiseType == 'realNoise':
				self.realNoise = True
				self.crossTalk = False
				# self.realNoise = realNoise
				self.realNoiseChannelTensor = None
				self.realNoiseChannelTensor = czExp_channel(filename=chiFilename)
			#
			elif noiseType == 'idealNoise':
				self.idealNoise = True
			else:
				raise TypeError(f'Noise type "{noiseType}" is not supported yet.')

		# Paras. About Torch
		self.device = select_device(device)
		self.dtype = tc.complex128
		self.layers = nn.Sequential()
		self._oqs_list = []

		# Density Matrix
		self.DM = None
		self.DMNode = None

		# TNN Truncation
		self.chi = chi
		self.kappa = kappa
		self.tnn_optimize = tnn_optimize

	def _transpile_gate(self, _gate_: AbstractGate, _oqs_: list[int]):
		_gateName_ = _gate_.name.lower()
		if _gateName_ == 'cnot' or _gateName_ == 'cx':
			_gateList_ = [AbstractGate(ideal=True).ry(-np.pi / 2),
			              AbstractGate().czEXP(EXPTensor=self.realNoiseChannelTensor),
			              AbstractGate(ideal=True).ry(np.pi / 2)]
			_oqsList_ = [[_oqs_[-1]], _oqs_, [_oqs_[-1]]]
		elif _gateName_ == 'rzz':
			_gateList_ = [AbstractGate().cnot(), AbstractGate(ideal=True).rz(_gate_.para), AbstractGate().cnot()]
			_oqsList_ = [_oqs_, [_oqs_[-1]], _oqs_]
		elif _gateName_ == 'rxx':
			_gateList_ = [AbstractGate(ideal=True).h(), AbstractGate().cnot(), AbstractGate(ideal=True).rz(_gate_.para),
			              AbstractGate().cnot(), AbstractGate(ideal=True).h()]
			_oqsList_ = [_oqs_, _oqs_, [_oqs_[-1]], _oqs_, _oqs_]
		elif _gateName_ == 'ryy':
			_gateList_ = [AbstractGate(ideal=True).rx(np.pi / 2), AbstractGate().cnot(),
			              AbstractGate(ideal=True).rz(_gate_.para), AbstractGate().cnot(),
			              AbstractGate(ideal=True).rx(-np.pi / 2)]
			_oqsList_ = [_oqs_, _oqs_, [_oqs_[-1]], _oqs_, _oqs_]
		else:
			_gateList_, _oqsList_ = [_gate_], [_oqs_]
		return _gateList_, _oqsList_

	def _crossTalkZ_transpile(self, _gate_: AbstractGate, _oqs_: list[int]):
		_minOqs, _maxOqs = min(_oqs_), max(_oqs_)
		if _minOqs == _maxOqs:
			_Angle = np.random.normal(loc=np.pi / 16, scale=np.pi / 128,
			                          size=(1, 2))  # Should be related to the chip-Exp information
			_gateList_ = [AbstractGate(ideal=True).rz(_Angle[0][0]), _gate_, AbstractGate(ideal=True).rz(_Angle[0][1])]
			_oqsList_ = [[_oqs_[0] - 1], _oqs_, [_oqs_[0] + 1]]
		elif _minOqs + 1 == _maxOqs:
			_Angle = np.random.normal(loc=np.pi / 16, scale=np.pi / 128,
			                          size=(1, 2))  # Should be related to the chip-Exp information
			_gateList_ = [AbstractGate(ideal=True).rz(_Angle[0][0]), _gate_, AbstractGate(ideal=True).rz(_Angle[0][1])]
			_oqsList_ = [[_minOqs - 1], _oqs_, [_maxOqs + 1]]
		else:
			_Angle = np.random.normal(loc=np.pi / 16, scale=np.pi / 128,
			                          size=(1, 4))  # Should be related to the chip-Exp information
			_gateList_ = [AbstractGate(ideal=True).rz(_Angle[0][0]), AbstractGate(ideal=True).rz(_Angle[0][1]),
			              _gate_, AbstractGate(ideal=True).rz(_Angle[0][2]), AbstractGate(ideal=True).rz(_Angle[0][3])]
			_oqsList_ = [[_minOqs - 1], [_minOqs + 1], _oqs_, [_maxOqs - 1], [_maxOqs + 1]]
		if _oqsList_[0][0] < 0:
			_gateList_.pop(0), _oqsList_.pop(0)
		if _oqsList_[-1][0] > self.qnumber - 1:
			_gateList_.pop(-1), _oqsList_.pop(-1)
		return _gateList_, _oqsList_

	def _add_gate(self, _qubits: list[tn.Node] or list[tn.AbstractNode],
	              _layer_num: int, _oqs: list[int]):
		r"""
		Add quantum Gate to tensornetwork

		Args:
			_oqs: operating qubits.

		Returns:
			Tensornetwork after adding gate.
		"""

		gate = self.layers[_layer_num].gate
		_maxIdx, _minIdx = max(_oqs), min(_oqs)

		if not isinstance(_qubits, list):
			raise TypeError('Qubit must be a list of nodes.')
		if not isinstance(gate, (TensorGate, NoisyTensorGate)):
			raise TypeError(f'Gate must be a TensorGate, current type is {type(gate)}.')
		if not isinstance(_oqs, list):
			raise TypeError('Operating qubits must be a list.')
		if _maxIdx >= self.qnumber:
			raise ValueError('Qubit index out of range.')

		single = gate.single

		if single is False:  # Two-qubit gate
			"""
						   | l  | p                          | j              | w
						___|____|___                    ____|____        ____|____
						|          |                    |       |        |       |
						|    DG    |---- q        l ----| Qubit |--------| Qubit |----v
						|__________|                    |_______|    k   |_______|
						   |    |                           |                |
						   | j  | w                         | m              | n
			"""
			if len(_oqs) != 2 or _minIdx == _maxIdx:
				raise ValueError('Invalid operating qubits for a two-qubit gate.')
			if is_nested(_oqs):
				raise NotImplementedError('Series CNOT gates are not supported yet.')

			_gateAxisNames = [f'physics_{_minIdx}', f'physics_{_maxIdx}', f'inner_{_minIdx}', f'inner_{_maxIdx}']
			if self.realNoise or self.idealNoise:
				if not self.realNoise:
					# Adding depolarization noise channel to Two-qubit gate
					gate.tensor = tc.einsum('ijklp, klmn -> ijmnp', self.Noise.dpCTensor2, gate.tensor)
				if gate.tensor.shape[-1] != 1 or len(gate.tensor.shape) != 4:
					_gateAxisNames.append(f'I_{_minIdx}')

			gate = tn.Node(gate.tensor.squeeze(), name=gate.name, axis_names=_gateAxisNames)

			# Created a new node in memory
			_contract_qubits = tn.contract_between(_qubits[_minIdx], _qubits[_maxIdx],
			                                       name=f'q_{_oqs[0]}_{_oqs[1]}',
			                                       allow_outer_product=True)
			# ProcessFunction, for details, see the function definition.
			EdgeName2AxisName([_contract_qubits])

			_contract_qubitsAxisName = _contract_qubits.axis_names

			_gString = 'ijwpq' if len(gate.tensor.shape) == 5 else 'ijwp'
			if _oqs[0] > _oqs[1]:
				_gString = _gString.replace('wp', 'pw')

			_smallI, _bigI, _lBond, _rBond = f'I_{min(_oqs)}' in _contract_qubitsAxisName,\
				f'I_{max(_oqs)}' in _contract_qubitsAxisName,\
				'bond' in _contract_qubitsAxisName[0], \
				'bond' in _contract_qubitsAxisName[-1]

			_qString = ''.join(['l' * _lBond, 'wp', 'm' * _smallI, 'n' * _bigI, 'r' * _rBond])

			_qAFString = _qString.replace('wp', _gString[:2] + _gString[-1] * (len(gate.tensor.shape) == 5))
			if _oqs[0] > _oqs[1]:
				_qAFString = _qAFString.replace('ij', 'ji')

			_reorderAxisName = [f'bond_{_minIdx - 1}_{_minIdx}'] * _lBond \
			                   + [f'physics_{_minIdx}', f'physics_{_maxIdx}'] \
			                   + [f'I_{_minIdx}'] * _smallI + [f'I_{_maxIdx}'] * _bigI \
			                   + [f'bond_{_maxIdx}_{_maxIdx + 1}'] * _rBond

			_contract_qubits.reorder_edges([_contract_qubits[_element] for _element in _reorderAxisName])
			_contract_qubitsTensor_AoP = tc.einsum(f'{_gString}, {_qString} -> {_qAFString}',
			                                       gate.tensor, _contract_qubits.tensor)
			_qShape = _contract_qubitsTensor_AoP.shape

			_mIdx, _qIdx = _qAFString.find('m'), _qAFString.find('n')
			if _mIdx != -1:     # qubit[min] is Noisy before this gate-add operation
				_qShape = _qShape[:_mIdx - 1] + (_qShape[_mIdx - 1] * _qShape[_mIdx],) + _qShape[_mIdx + 1:]
				_contract_qubits.set_tensor(tc.reshape(_contract_qubitsTensor_AoP, _qShape))
			else:
				print(_contract_qubits.edges)
				_contract_qubits.set_tensor(_contract_qubitsTensor_AoP)
				if 'q' in _gString:
					_contract_qubitsAxisName.insert(
						_contract_qubitsAxisName.index(f'physics_{_maxIdx}') + 1, f'I_{_minIdx}'
					)
					_contract_qubits.add_axis_names(_contract_qubitsAxisName)
					_new_edge = tn.Edge(_contract_qubits,
					                    axis1=_contract_qubitsAxisName.index(f'I_{_minIdx}'), name=f'I_{_minIdx}')
					_contract_qubits.edges.insert(_contract_qubitsAxisName.index(f'I_{_minIdx}'), _new_edge)
					_contract_qubits.add_edge(_new_edge, f'I_{_minIdx}')

			print(_contract_qubits.edges)

			# Split back to two qubits
			_left_AxisName = ['bond_{}_{}'.format(_minIdx - 1, _minIdx)] * _lBond + [f'physics_{_minIdx}']
			_right_AxisName = [f'physics_{_maxIdx}'] + ['bond_{}_{}'.format(_maxIdx, _maxIdx + 1)] * _rBond
			if self.idealNoise or self.realNoise:
				_left_AxisName.append(f'I_{_minIdx}')
				if _bigI:
					_right_AxisName.append(f'I_{_maxIdx}')

			_left_edges, _right_edges = [_contract_qubits[name] for name in _left_AxisName], \
				[_contract_qubits[name] for name in _right_AxisName]

			_qubits[_minIdx], _qubits[_maxIdx], _ = tn.split_node(_contract_qubits,
			                                                      left_edges=_left_edges,
			                                                      right_edges=_right_edges,
			                                                      left_name=f'qubit_{_minIdx}',
			                                                      right_name=f'qubit_{_maxIdx}',
			                                                      edge_name=f'bond_{_minIdx}_{_maxIdx}')
			EdgeName2AxisName([_qubits[_minIdx], _qubits[_minIdx]])

			if (self.realNoise or self.idealNoise) and f'I_{_oqs[1]}' in _qubits[_oqs[1]].axis_names:
				# Shape-relating
				_shape = _qubits[_oqs[1]].tensor.shape
				_left_edge_shape = [_shape[_ii_] for _ii_ in range(len(_shape) - 1)]
				_left_dim = int(np.prod(_left_edge_shape))
				# SVD to truncate the inner dimension
				_u, _s, _ = tc.linalg.svd(tc.reshape(_qubits[_oqs[1]].tensor, (_left_dim, _shape[-1])),
				                          full_matrices=False)
				_s = _s.to(dtype=tc.complex128)

				# Truncate the inner dimension
				if self.kappa is None:
					_kappa = _s.nelement()
				else:
					_kappa = self.kappa

				_s = _s[: _kappa]
				_u = _u[:, : _kappa]

				if len(_s.shape) == 1:
					_s = tc.diag(_s)

				# Back to the former shape
				_left_edge_shape.append(_s.shape[-1])
				_qubits[_oqs[1]].tensor = tc.reshape(tc.matmul(_u, _s), _left_edge_shape)
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
			gate_list = [
				tn.Node(
					tc.reshape(
						tc.einsum('ij, jkl, kmn -> imln', gate.tensor, self.Noise.dpCTensor, self.Noise.apdeCTensor),
						(2, 2, -1))
					if (self.idealNoise or self.unified) and not gate.ideal  # Noise channel is added ONLY when the cases idealNoise/Unified
					else gate.tensor,
					name=gate.name,
					axis_names=[f'physics_{_idx}', f'inner_{_idx}', f'I_{_idx}'] if self.idealNoise or self.unified else [
						f'physics_{_idx}', f'inner_{_idx}']
				)
				for _idx in _oqs
			]

			for _i, _bit in enumerate(_oqs):
				_qubit = _qubits[_bit]
				_qubitTensor, _qubitAxisName = _qubit.tensor, _qubit.axis_names

				_qString = ''.join([
					'l' * ('bond' in _qubitAxisName[0]),
					'i',
					'j' * (f'I_{_bit}' in _qubitAxisName),
					'r' * ('bond' in _qubitAxisName[-1])
				])
				_gString = 'min' if len(gate_list[_i].shape) == 3 else 'mi'

				_qAFString = _qString.replace('i', 'mn') if _gString == 'min' else _qString.replace('i', 'm')
				_qubitTensor_AOP = tc.einsum(f'{_gString}, {_qString} -> {_qAFString}', gate_list[_i].tensor, _qubitTensor)
				_qShape = _qubitTensor_AOP.shape

				_jIdx, _nIdx = _qAFString.find('j'), _qAFString.find('n')
				if _jIdx != -1 and _nIdx != -1:  # qubit is Noisy before this gate-add operation
					_qShape = _qShape[:_jIdx - 1] + (_qShape[_jIdx - 1] * _qShape[_jIdx], ) + _qShape[_jIdx + 1:]
					_qubit.tensor = tc.reshape(_qubitTensor_AOP, _qShape)
				else:
					_qubit.tensor = _qubitTensor_AOP
					if 'n' in _gString:
						_qubitAxisName.insert(_qubitAxisName.index(f'physics_{_bit}') + 1, f'I_{_bit}')
						_qubit.add_axis_names(_qubitAxisName)
						_new_edge = tn.Edge(_qubit, axis1=_qubitAxisName.index(f'I_{_bit}'), name=f'I_{_bit}')
						_qubit.edges.insert(_qubitAxisName.index(f'I_{_bit}'), _new_edge)
						_qubit.add_edge(_new_edge, f'I_{_bit}')

	def add_gate(self, gate: AbstractGate, oqs: list):
		r"""
		Add quantum gate to circuit layer by layer.

		Args:
			gate: gate to be added;
			oqs: operating qubits.

		Returns:
			None
		"""
		if not isinstance(gate, AbstractGate):
			raise TypeError('Gate must be a AbstractGate.')
		if isinstance(oqs, int):
			oqs = [oqs]
		if not isinstance(oqs, list):
			raise TypeError('Operating qubits must be a list.')

		if self.realNoise:
			_transpile_gateList, _transpile_oqsList = self._transpile_gate(gate, oqs)
		elif self.crossTalk:
			_transpile_gateList, _transpile_oqsList = self._crossTalkZ_transpile(gate, oqs)
		else:
			_transpile_gateList, _transpile_oqsList = [gate], [oqs]

		for _num, _gate in enumerate(_transpile_gateList):
			if _gate.para is None:
				_para = None
			else:
				_para = _gate.para
				if type(_para) is float:
					_para = tc.tensor(_para)
				_paraI, _paraD = '{:.3f}'.format(_para.squeeze()).split('.')
				_para = f'{_paraI}|{_paraD} pi'

			self.layers.add_module(f'{_gate.name}{_transpile_oqsList[_num]}({_para})-G{self.i}', _gate)
			self._oqs_list.append(_transpile_oqsList[_num])
		self.i += 1

	def _calculate_DM(self, state_vector: bool = False,
	                  reduced_index: Optional[list[list[int] or int]] = None) -> tc.Tensor:
		r"""
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

		def _re_permute(_axis_names_: list[str]):
			_left_, _right_ = [], []
			for _idx_, _name_ in enumerate(_axis_names_):
				if 'con_' not in _name_:
					_left_.append(_idx_)
				else:
					_right_.append(_idx_)
			return _left_ + _right_

		if not reduced_index:
			reduced_index = None

		if reduced_index is not None:
			if reduced_index[np.argmax(reduced_index)] >= self.qnumber:
				raise ValueError('Reduced index should not be larger than the qubit number.')

		if not state_vector:
			_qubits_conj = deepcopy(self.state)     # Node.copy() func. cannot work correctly
			for _qubit in _qubits_conj:
				_qubit.set_tensor(_qubit.tensor.conj())

			# Differential name the conjugate qubits' edges name to permute the order of the indices
			for _i, _qubit_conj in enumerate(_qubits_conj):
				_qubit_conj.set_name(f'con_{_qubit_conj.name}')
				for _ii in range(len(_qubit_conj.edges)):
					if 'physics' in _qubit_conj[_ii].name:
						_qubit_conj[_ii].set_name(f'con_{_qubit_conj[_ii].name}')
						_qubit_conj.axis_names[_ii] = f'con_{_qubit_conj.axis_names[_ii]}'

			for i in range(len(self.state)):
				if not self.ideal:
					if f'I_{i}' in self.state[i].axis_names:
						tn.connect(self.state[i][f'I_{i}'], _qubits_conj[i][f'I_{i}'])

			# Reduced density matrix
			if reduced_index is not None:
				if isinstance(reduced_index, int):
					reduced_index = [reduced_index]
				if not isinstance(reduced_index, list):
					raise TypeError('reduced_index should be int or list[int]')
				for _idx in reduced_index:
					tn.connect(self.state[_idx][f'physics_{_idx}'],
					           _qubits_conj[_idx][f'con_physics_{_idx}'])
			else:
				reduced_index = []

			_numList = [_i for _i in range(self.qnumber) if _i not in reduced_index]
			_qOutOrder, _conQOutOrder = [self.state[i][f'physics_{i}'] for i in _numList],\
				[_qubits_conj[i][f'con_physics_{i}'] for i in _numList]

			_dm = tn.contractors.auto(self.state + _qubits_conj,
			                          output_edge_order=_qOutOrder + _conQOutOrder)

			_reshape_size = self.qnumber - len(reduced_index)

			self.DM, self.DMNode = _dm.tensor.reshape((2 ** _reshape_size, 2 ** _reshape_size)), self.DMNode
			return self.DM
		else:
			if self.fVR is False:
				if self.ideal is False:
					raise ValueError('Noisy circuit cannot be represented by state vector efficiently.')
			if reduced_index is not None:
				raise ValueError('State vector cannot efficiently represents the reduced density matrix.')

			_outOrder = [self.state[i][f'physics_{i}'] for i in list(range(self.qnumber))]
			_vector = tn.contractors.auto(self.state, output_edge_order=_outOrder)

			if self.fVR is False:
				self.DM = _vector.tensor.reshape((2 ** self.qnumber, 1))
			else:
				self.DM = _vector.tensor

			return self.DM

	def forward(self, _state: list[tn.Node] = None,
	            state_vector: bool = False, reduced_index: list = None, forceVectorRequire: bool = False) -> tc.Tensor:
		r"""
		Forward propagation of tensornetwork.

		Returns:
			self.state: tensornetwork after forward propagation.
		"""
		self.initState = _state
		self.qnumber = len(_state)
		self.fVR = forceVectorRequire

		for _i in range(len(self.layers)):
			self._add_gate(_state, _i, _oqs=self._oqs_list[_i])
			if self.layers[_i]._lastTruncation and self.tnn_optimize is True:
				check = checkConnectivity(_state)
				if check is True:
					qr_left2right(_state)
					svd_right2left(_state, chi=self.chi)
		self.state = _state
		_dm = self._calculate_DM(state_vector=state_vector, reduced_index=reduced_index)

		return _dm
