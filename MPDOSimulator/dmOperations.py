"""
Author: Weiguo Ma
Time: 11.27.2024
Contact: weiguo.m@iphy.ac.cn
"""
from typing import Optional, List, Union

import tensornetwork as tn
from torch import Tensor


def reduce_dmNodes(qubits_nodes: List[tn.AbstractNode],
                   conj_qubits_nodes: List[tn.AbstractNode],
                   reduced_index: Optional[List[Union[List[int], int]]] = None):
    """
    Reduce nodes in memory.
    """
    if reduced_index is None:
        return None
    if reduced_index and max(reduced_index) >= len(qubits_nodes):
        raise ValueError('Reduced index should not be larger than the qubit number.')

    # Reduced density matrix
    if reduced_index:
        reduced_index = [reduced_index] if isinstance(reduced_index, int) else reduced_index
        if not isinstance(reduced_index, list):
            raise TypeError('reduced_index should be int or list[int]')
        for _idx in reduced_index:
            tn.connect(qubits_nodes[_idx][f'physics_{_idx}'], conj_qubits_nodes[_idx][f'con_physics_{_idx}'])


def trace_rho(dmNodes: List[tn.AbstractNode]) -> Tensor:
    _qNum = len(dmNodes) // 2
    _dmNodes = tn.replicate_nodes(dmNodes)
    _qubits, _qubits_conj = _dmNodes[:_qNum], _dmNodes[_qNum:]

    for i in range(_qNum):
        if _qubits[i][f'physics_{i}'].is_dangling():
            _qubits[i][f'physics_{i}'] ^ _qubits_conj[i][f'con_physics_{i}']
    return tn.contractors.auto(_dmNodes).tensor.real


def trace_rho_rho(dmNodes_0: List[tn.AbstractNode], dmNodes_1: Optional[List[tn.AbstractNode]] = None) -> Tensor:
    _qNum = len(dmNodes_0) // 2
    if dmNodes_1 is not None and len(dmNodes_1) // 2 != _qNum:
        raise ValueError('Density matrices should have same number of nodes.')

    _all_nodes = []
    with tn.NodeCollection(_all_nodes):
        _dmNodes_0 = tn.replicate_nodes(dmNodes_0)
        _dmNodes_1 = tn.replicate_nodes(dmNodes_1) if dmNodes_1 is not None else tn.replicate_nodes(dmNodes_0)
        for idx in range(_qNum):
            if _dmNodes_0[idx + _qNum][f'con_physics_{idx}'].is_dangling():
                _dmNodes_0[idx + _qNum][f'con_physics_{idx}'] ^ _dmNodes_1[idx][f'physics_{idx}']
            if _dmNodes_0[idx][f'physics_{idx}'].is_dangling():
                _dmNodes_0[idx][f'physics_{idx}'] ^ _dmNodes_1[idx + _qNum][f'con_physics_{idx}']

    return tn.contractors.auto(_all_nodes).tensor.real


def trace_rho2(dmNodes: List[tn.AbstractNode]) -> Tensor:
    return trace_rho_rho(dmNodes)


def trace_composited_rho(*dmNodes: List[tn.AbstractNode]) -> Tensor:
    r"""
    If \rho = (\rho_0 + \rho_1 + \rho_2) / slices,
        then \trace(\rho) = 1 / slices * \trace(\rho_0 + \rho_1 + \rho_2).
    Args:
        *dmNodes: Density matrix nodes list before contraction.

    Returns:
        Trace value in float.
    """
    _sum = 0
    for _dmNode in dmNodes:
        _sum += trace_rho_rho(_dmNode)

    return _sum


def trace_composited_rho2(*dmNodes: List[tn.AbstractNode]) -> Tensor:
    r"""
    For \trace(\rho^2) = 1 / slices^2 * \trace[(\rho_0 + \rho_1 + \rho_2)(\rho_0 + \rho_1 + \rho_2)],
        that is 1 / slices^2 *
         \trace[
            \rho_0^2 + \rho_1^2 + \rho_2^2 +
             \rho_0\rho_1 + \rho_1\rho_0 + \rho_1\rho_2 +
              \rho_2\rho_1 + \rho_0\rho_2 + \rho_2\rho_0
              ]
    Args:
        *dmNodes: Density matrix nodes list before contraction.

    Returns:
        Trace value in float.
    """
    _sliceNum = len(dmNodes)

    _sum = 0
    for _i in range(_sliceNum):
        for _j in range(_sliceNum):
            _sum += trace_rho_rho(dmNodes[_i], dmNodes[_j])

    return _sum / (_sliceNum ** 2)


def expect(dmNodes: List[tn.AbstractNode],
           observables: Union[Tensor, List[Tensor]], oqs: Union[int, List]) -> Union[List[Tensor], Tensor]:
    qnumber = len(dmNodes) // 2
    oqs = [oqs] if isinstance(oqs, int) else oqs
    observables = [observables] if isinstance(observables, Tensor) else observables

    expectation_values = [Tensor(0)] * len(observables)
    for j, (obs, oq) in enumerate(zip(observables, oqs)):
        oq = [oq] if isinstance(oq, int) else oq
        _all_nodes = []
        with tn.NodeCollection(_all_nodes):
            obs_dim, oq_dim = obs.dim(), len(oq)
            try:
                obs = obs.reshape([2] * 2 * oq_dim)
            except RuntimeError:
                raise ValueError(f'Shape of the No.{j} obs is not valid, which is: {obs.shape}.')

            if obs_dim > qnumber or oq_dim > qnumber:
                raise ValueError(f'Dim of No.{j} - oqs: {obs_dim} or obs: {oq_dim} exceeds the system size.')
            if obs_dim != oq_dim:
                raise ValueError(f'Dim of No.{j} - oqs and obs do not match.')

            _axis_names = [f'physics_{_i}' for _i in oq] + [f'con_physics_{_i}' for _i in oq]
            _obs_node = tn.Node(obs, name=f'obs_{oq}', axis_names=_axis_names)

            _dmNodes = tn.replicate_nodes(dmNodes)
            for _idx in range(qnumber):
                try:
                    _obs_node[f'con_physics_{_idx}'] ^ _dmNodes[_idx][f'physics_{_idx}']
                    _obs_node[f'physics_{_idx}'] ^ _dmNodes[_idx + qnumber][f'con_physics_{_idx}']
                except ValueError:
                    _dmNodes[_idx][f'physics_{_idx}'] ^ _dmNodes[_idx + qnumber][f'con_physics_{_idx}']

        expectation_values[j] = tn.contractors.auto(_all_nodes).tensor.real

    return expectation_values
