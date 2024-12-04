from torch import complex64
import tensornetwork as tn
import MPDOSimulator as Simulator

tn.set_default_backend("pytorch")

DTYPE, DEVICE = complex64, 'cpu'

# Basic information of circuit
qnumber = 5
ideal_circuit = False  # or True
noiseType = 'realNoise'  # or 'realNoise' or 'idealNoise'

chiFileNames = {
    'CZ': {'01': './MPDOSimulator/chi/czDefault.mat', '12': './MPDOSimulator/chi/czDefault.mat', '23': './MPDOSimulator/chi/czDefault.mat',
           '34': './MPDOSimulator/chi/czDefault.mat'},
    'CP': {}
}
chi, kappa = 4, 4

# Establish a quantum circuit
circuit = Simulator.TensorCircuit(qn=qnumber, ideal=ideal_circuit, noiseType=noiseType,
                        chiFileDict=chiFileNames, chi=chi, kappa=kappa, chip='beta4Test', dtype=DTYPE, device=DEVICE)

# Test 1
# circuit.y(1)
# circuit.h(2)
# circuit.cnot(1, 0)
# circuit.x(0)
# circuit.z(1)
# circuit.y(1)
# circuit.cnot(2, 1)

# GHZ TEST
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)
circuit.cnot(2, 3)
circuit.cnot(3, 4)


# Set TensorNetwork Truncation
circuit.truncate()

print(circuit)

# Generate an initial quantum state
state = Simulator.Tools.create_ket0Series(qnumber, dtype=DTYPE, device=DEVICE)

# Evolve the circuit with given state, same circuit can be reused.
circuit.evolve(state)

# Function 1: Sample from the given circuit without getting density matrix.
    # This could be slower for small system than getting its full density matrix.
counts = circuit.fakeSample(1024)[-1]       # This function returns (measurement_outcomes, counts_statistical)
print(counts)

## Plot the counts
# Simulator.Tools.plot_histogram(
#     counts, title=f'"{noiseType}" Probability Distribution', filename='./figs/test.pdf', show=False
# )

# Function 2: Get state node (tensor network without contraction):
# state_nodes = circuit.stateNodes

# Function 3: Get density matrix (contracted):
    # This function returns the density matrix of the quantum system, which could cost much memory for a huge system.
    # After the calculation, you could call circuit.dm to get this property.
# dm = circuit.cal_dm()

# Function 4: Get density matrix nodes (tensor network without contraction):
    # After the calculation, you could call circuit.dmNodes to get this property.
# dmNodes = circuit.cal_dmNodes()

# Function 5: Get state vector (contracted):
    # While you are simulating the IDEAL quantum circuit, you could call this to get a \ket state matrix.
    # After the calculation, you could call circuit.vector to get this property.
# vector = circuit.cal_vector()

# Function 6: Calculate the expectation values with density matrix nodes (tensor network without contraction):
# obs = [
#     kron(tensor([[0, 1], [1, 0]], dtype=DTYPE, device=DEVICE), tensor([[0, 1], [1, 0]], dtype=DTYPE, device=DEVICE)),
#     kron(tensor([[0, -1j], [1j, 0]], dtype=DTYPE, device=DEVICE), tensor([[0, -1j], [1j, 0]], dtype=DTYPE, device=DEVICE)),
#     kron(tensor([[1, 0], [0, -1]], dtype=DTYPE, device=DEVICE), tensor([[1, 0], [0, -1]], dtype=DTYPE, device=DEVICE)),
#     1 / 2 * kron(tensor([[1, 1], [1, -1]], dtype=DTYPE, device=DEVICE), tensor([[1, 1], [1, -1]], dtype=DTYPE, device=DEVICE)),
# ]
# print(Simulator.dmOperations.expect(dmNodes, obs, oqs=[[0, 1], [0, 1], [0, 1], [0, 1]]))

# Function 7: Pauli Measurement:
    # I provide two methods of Pauli measurements:
    # 1. circuit.measure(oqs, orientation) -> This call() could show density matrix and dmNodes and samples.
    # 2. circuit.fakeSample(shots, orientation, reduced) -> This call() show only samples. reduced: allow to sample from reduced system.


# To reuse all the property in Nodes,
# I recommend you to use tensornetwork.replicate_nodes() for keeping the original nodes, which are in memory, unchanged.