import tensornetwork as tn

import MPDOSimulator as Simulator

tn.set_default_backend("pytorch")

# Basic information of circuit
qnumber = 4
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
                        chiFileDict=chiFileNames, chi=chi, kappa=kappa, chip='beta4Test', device='cpu')

circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)
circuit.cnot(2, 3)

# Set TensorNetwork Truncation
circuit.truncate()

print(circuit)

# Generate an initial quantum state
state = Simulator.Tools.create_ket0Series(qnumber)
state = circuit(state, state_vector=False, reduced_index=[])
# print(state)

# Calculate probability distribution
prob_dict = Simulator.Tools.density2prob(state, tol=5e-4)  # Set _dict=False to return a np.array
print(prob_dict)

# plot probability distribution
Simulator.Tools.plot_histogram(prob_dict, title=f'"{noiseType}" Probability Distribution', filename='./figs/test.pdf', show=False)
