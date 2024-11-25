"""
Author: weiguo_ma
Time: 04.13.2023
Contact: weiguo.m@iphy.ac.cn
"""
from typing import List, Union, Optional, cast

import tensornetwork as tn
import torch as tc

from .Tools import EdgeName2AxisName

__all__ = [
    'qr_left2right',
    'svd_left2right',
    'svd_right2left',
    'svdKappa_left2right',
    'bondTruncate',
    'checkConnectivity'
]


def _randomized_svd(M, n_components: int, n_overSamples: int = 5,
                    n_iter: Union[str, int] = 'auto', random_state: Optional = None):
    """
    Randomized SVD for complex-number matrix.
    """
    _m, _n = M.shape
    _rng = tc.Generator()
    if random_state is not None:
        _rng.manual_seed(random_state)

    _Q = tc.randn(_m, n_components + n_overSamples, dtype=tc.complex64, generator=_rng)

    if n_iter == 'auto':
        n_iter = 6 if _m >= _n else 4

    for _ in range(n_iter):
        _Q = M @ (M.T.conj() @ _Q)

    _Q, _ = tc.linalg.qr(_Q)

    _B = _Q.T.conj() @ M

    _u, _s, _vh = tc.linalg.svd(_B, full_matrices=False)
    _u = _Q @ _u

    return _u, _s, _vh


def checkConnectivity(_qubits: Union[List[tn.Node], List[tn.AbstractNode]]):
    """
    Check if the qubits have connectivity.

    Args:
        _qubits: List of qubit nodes.

    Returns:
        True if qubits are connected, False otherwise.
    """
    if len(_qubits) <= 1:
        return True
    try:
        tn.check_connected(_qubits)
        return True
    except ValueError:
        return False


def bondTruncate(_qubits: Union[List[tn.Node], List[tn.AbstractNode]], chi: int, regularization: bool = False):
    """
    Call of qr -> svd.

    Args:
        _qubits: List of qubits (as nodes).
        chi: bond dimension.
        regularization: True for regularization.

    Returns:
        None: The function modifies the input tensors in-place.
    """
    if chi is not None or regularization:
        qr_left2right(_qubits)
        svd_right2left(_qubits, chi=chi)


def qr_left2right(_qubits: Union[List[tn.Node], List[tn.AbstractNode]]):
    """
    QR decomposition from left to right.

    Args:
        _qubits: List of qubits (as nodes).

    Returns:
        None: The function modifies the input tensors in-place.
    """
    if not isinstance(_qubits, List):
        raise TypeError('input should be a list of qubits nodes')

    for _i in range(len(_qubits) - 1):
        _edges = _qubits[_i].edges
        _left_edges, _right_edges = (
            [_edge for _edge in _edges if _edge.name != f'bond_{_i}_{_i + 1}'],
            [_qubits[_i][f'bond_{_i}_{_i + 1}']]
        )

        _q, _r = tn.split_node_qr(_qubits[_i],
                                  left_edges=_left_edges,
                                  right_edges=_right_edges,
                                  left_name=_qubits[_i].name,
                                  right_name='right_waiting4contract2form_right',
                                  edge_name=f'qrbond_{_i}_{_i + 1}')

        _r = tn.contract_between(_r, _qubits[_i + 1], name=_qubits[_i + 1].name)
        _qubits[_i], _qubits[_i + 1] = _q, _r

        EdgeName2AxisName([_qubits[_i], _qubits[_i + 1]])


def svd_right2left(_qubits: Union[List[tn.Node], List[tn.AbstractNode]], chi: int):
    r"""
    SVD from right to left.

    Args:
        _qubits: list of nodes;
        chi: bond dimension.

    Returns:
        None: The function modifies the input tensors in-place.
    """
    if not isinstance(_qubits, List):
        raise TypeError('input should be a list of qubits nodes')

    for idx in range(len(_qubits) - 1, 0, -1):
        _left_edges = [name for name in _qubits[idx - 1].axis_names if name not in _qubits[idx].axis_names]
        _right_edges = [name for name in _qubits[idx].axis_names if name not in _qubits[idx - 1].axis_names]

        _left_edges = [_qubits[idx - 1][_left_name] for _left_name in _left_edges]
        _right_edges = [_qubits[idx][_right_name] for _right_name in _right_edges]

        contracted_two_nodes = tn.contract_between(_qubits[idx - 1], _qubits[idx], name='contract_two_nodes')
        EdgeName2AxisName([contracted_two_nodes])

        _qubits[idx - 1], _qubits[idx], _ = tn.split_node(contracted_two_nodes,
                                                          left_edges=_left_edges,
                                                          right_edges=_right_edges,
                                                          left_name=_qubits[idx - 1].name,
                                                          right_name=_qubits[idx].name,
                                                          edge_name=f'bond_{idx - 1}_{idx}',
                                                          max_singular_values=chi)
        EdgeName2AxisName([_qubits[idx - 1], _qubits[idx]])


def svd_left2right(_qubits: Union[List[tn.Node], List[tn.AbstractNode]], chi: int):
    """
        SVD from left to right.

        Args:
            _qubits: list of nodes;
            chi: bond dimension.

        Returns:
            None: The function modifies the input tensors in-place.
        """
    if not isinstance(_qubits, List):
        raise TypeError('input should be a list of qubits nodes')

    for idx in range(len(_qubits) - 1):
        _left_edges = [name for name in _qubits[idx].axis_names if name not in _qubits[idx + 1].axis_names]
        _right_edges = [name for name in _qubits[idx + 1].axis_names if name not in _qubits[idx].axis_names]

        _left_edges = [_qubits[idx][_left_name] for _left_name in _left_edges]
        _right_edges = [_qubits[idx + 1][_right_name] for _right_name in _right_edges]

        contracted_two_nodes = tn.contract_between(_qubits[idx], _qubits[idx + 1], name='contract_two_nodes')
        EdgeName2AxisName([contracted_two_nodes])

        _qubits[idx], _qubits[idx + 1], _ = tn.split_node(contracted_two_nodes,
                                                          left_edges=_left_edges,
                                                          right_edges=_right_edges,
                                                          left_name=_qubits[idx].name,
                                                          right_name=_qubits[idx + 1].name,
                                                          edge_name=f'bond_{idx}_{idx + 1}',
                                                          max_singular_values=chi)
        EdgeName2AxisName([_qubits[idx], _qubits[idx + 1]])


def svdKappa_left2right(_qubits: List[tn.AbstractNode], kappa: int):
    """
    Perform SVD with optional dimension truncation on a list of quantum tensors.

    Args:
        _qubits (list[tn.Node] or list[tn.AbstractNode]): List of quantum tensors.
        kappa (int, optional): The truncation dimension. If None, no truncation is performed.

    Returns:
        None: The function modifies the input tensors in-place.
    """
    if kappa is None:
        return None

    for _idx, _qubit in enumerate(_qubits):
        if not f'I_{_idx}' in _qubit.axis_names:
            continue
        if cast(int, _qubit[f'I_{_idx}'].dimension) <= kappa:
            continue

        _noiseEdge = [_qubit[f'I_{_idx}']]
        _otherEdges = [edge for edge in _qubit.edges if edge != _noiseEdge[0]]

        _U, _S, _, _ = tn.split_node_full_svd(
            _qubit, left_edges=_otherEdges, right_edges=_noiseEdge, max_singular_values=kappa,
            left_name=f'qubit_{_idx}', right_edge_name=f'kpI_{_idx}'
        )
        _U = tn.contract_between(_U, _S, name=_qubit.name)
        EdgeName2AxisName([_U])

        _U[f'I_{_idx}'].disconnect()
        _qubits[_idx] = _U


def cal_entropy(qubits: Union[List[tn.Node], List[tn.AbstractNode]]):
    """
    Calculate entropy of a list of quantum tensors.

    Args:
        qubits (list[tn.Node] or list[tn.AbstractNode]): List of quantum tensors.
        kappa (int, optional): The truncation dimension. If None, no truncation is performed.

    Returns:
        _entropy: The function modifies the input tensors in-place.
    """
    raise NotImplementedError
