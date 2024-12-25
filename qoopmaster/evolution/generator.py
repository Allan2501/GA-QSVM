import numpy as np
import qiskit
import random
from ..backend import constant
from .environment_parent import Metadata
from .environment_synthesis import MetadataSynthesis
def initialize_random_parameters(num_qubits: int, max_operands: int, conditional: bool, seed):
    if max_operands < 1 or max_operands > 3:
        raise qiskit.circuit.exceptions.CircuitError("max_operands must be between 1 and 3")

    qr = qiskit.circuit.QuantumRegister(num_qubits, 'q')
    qc = qiskit.circuit.QuantumCircuit(num_qubits)

    if conditional:
        cr = qiskit.circuit.ClassicalRegister(num_qubits, 'c')
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)

    rng = np.random.default_rng(seed)
    thetas = qiskit.circuit.ParameterVector('theta')
    return qr, qc, rng, thetas
    
    
def choice_from_array(arr, condition):
    item = None
    while item is None:
        item = random.choice(arr)
        if condition(item):
            return item
        else:
            item = None
    return item



def by_depth_new(metadata: Metadata) -> qiskit.QuantumCircuit:
    num_qubits = metadata.num_qubits
    depth = metadata.num_circuit  # Assuming depth corresponds to num_circuit
    pool = constant.operations
    conditional = False
    seed = None
    max_operands = 2

    while True:  # Lặp đến khi tìm được mạch thỏa mãn điều kiện
        qr, qc, rng, thetas = initialize_random_parameters(num_qubits, max_operands, conditional, seed)
        thetas_length = 0
        for _ in range(depth):
            remaining_qubits = list(range(num_qubits))
            while remaining_qubits:
                max_possible_operands = min(len(remaining_qubits), max_operands)
                num_operands = choice_from_array(
                    [1, 1, 1, 1, 1, 1, 2, 2, 2, 2], lambda value: value <= max_possible_operands)
                rng.shuffle(remaining_qubits)
                operands = remaining_qubits[:num_operands]
                remaining_qubits = [
                    q for q in remaining_qubits if q not in operands]
                num_op_pool = [
                    item for item in pool if item['num_op'] == num_operands]

                operation = rng.choice(num_op_pool)
                num_params = operation['num_params']
                thetas_length += num_params
                thetas.resize(thetas_length)
                angles = thetas[thetas_length - num_params:thetas_length]
                register_operands = [qr[i] for i in operands]
                op = operation['operation'](*angles) if num_params > 0 else operation['operation']()
                qc.append(op, register_operands)

        # Kiểm tra điều kiện: len(qc.parameters) == qc.num_qubits
        if len(qc.parameters) == qc.num_qubits:
            break  # Nếu điều kiện thỏa mãn, thoát vòng lặp

    return qc



def by_depth(metadata: Metadata) -> qiskit.QuantumCircuit:
    num_qubits = metadata.num_qubits
    depth = metadata.depth
    pool = constant.operations
    conditional = False
    seed=None
    max_operands = 2
    qr, qc, rng, thetas = initialize_random_parameters(num_qubits, max_operands, conditional, seed)
    thetas_length = 0
    for _ in range(depth):
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = choice_from_array(
                [1, 1, 1, 1, 1, 1, 2, 2, 2, 2], lambda value: value <= max_possible_operands)
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands]
            remaining_qubits = [
                q for q in remaining_qubits if q not in operands]
            num_op_pool = [
                item for item in pool if item['num_op'] == num_operands]

            operation = rng.choice(num_op_pool)
            num_params = operation['num_params']
            thetas_length += num_params
            thetas.resize(thetas_length)
            angles = thetas[thetas_length - num_params:thetas_length]
            register_operands = [qr[i] for i in operands]
            op = operation['operation'](*angles)
            qc.append(op, register_operands)
    return qc

def weighted_choice(arr, weights):
    if len(arr) != len(weights):
        raise ValueError("The length of the arrays must be the same")
    if sum(weights) != 1:
        raise ValueError("The sum of the weights must be 1")
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return np.random.choice(arr, p=normalized_weights)

def generate_random_array(num_othergate, smallest_num_gate, n):
    # Adjust num_othergate to ensure it can be divided among n elements
    num_othergate_adjusted = num_othergate - n * smallest_num_gate
    
    if num_othergate_adjusted < 0:
        raise ValueError("num_othergate is too small to meet the condition with n elements.")
    
    # Generate an array of n elements, each initialized to smallest_num_gate
    arr = [smallest_num_gate] * n
    
    # Distribute the remaining sum (num_othergate_adjusted) randomly among the elements
    for i in range(n):
        arr[i] += random.randint(0, num_othergate_adjusted)
        num_othergate_adjusted -= arr[i] - smallest_num_gate
        
        if num_othergate_adjusted <= 0:
            break
    
    # Adjust to ensure the total sum is exactly num_othergate
    current_sum = sum(arr)
    while current_sum < num_othergate:
        arr[random.randint(0, n-1)] += 1
        current_sum += 1
    random.shuffle(arr)
    return arr





def by_num_cnot(metadata: MetadataSynthesis) -> qiskit.QuantumCircuit:
    """
    Generate a quantum circuit with a specified number of qubits, depth, and CNOT gates.
    The circuit ensures the total number of parameters from rx, ry, and rz gates equals num_qubits.

    Args:
        metadata (MetadataSynthesis): Contains the properties for the quantum circuit.

    Returns:
        qiskit.QuantumCircuit: Generated quantum circuit.
    """
    pool = constant.operations_only_cnot
    num_qubits = metadata.num_qubits
    depth = metadata.depth
    num_cnot = metadata.num_cnot
    num_gate = depth * num_qubits
    num_othergate = num_gate - num_cnot

    # Ensure total parameters from rx, ry, rz equals num_qubits
    num_params = num_qubits
    num_gates = [0] * len(pool)  # Initialize gate counts

    # Assign rx, ry, rz gates evenly to match num_params
    param_gates = ['rx', 'ry', 'rz']
    param_counts = [num_params // len(param_gates)] * len(param_gates)
    for i in range(num_params % len(param_gates)):
        param_counts[i] += 1

    # Add the determined number of rx, ry, rz gates to the pool
    gate_index_map = {gate['operation'].__name__: i for i, gate in enumerate(pool)}
    for gate_name, count in zip(param_gates, param_counts):
        num_gates[gate_index_map[gate_name]] = count

    # Distribute remaining gates (non-parameter gates)
    remaining_gates = num_othergate - num_params
    if remaining_gates > 0:
        smallest_num_gate = max(1, int(0.15 * remaining_gates))
        random_gates = generate_random_array(remaining_gates, smallest_num_gate, len(pool) - len(param_gates))
        for i, count in enumerate(random_gates):
            num_gates[i + len(param_gates)] = count

    # Add CNOT gates
    num_gates.append(num_cnot)

    # Create the circuit
    qc = qiskit.QuantumCircuit(num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta')
    thetas_length = 0
    full_pool = []

    # Populate the full_pool with gates based on num_gates
    for j, gate in enumerate(pool):
        full_pool.extend([gate] * num_gates[j])

    random.shuffle(full_pool)

    # Add gates to the circuit
    while len(full_pool) > 0:
        remaining_qubits = list(range(num_qubits))
        random.shuffle(remaining_qubits)

        while len(remaining_qubits) > 1:
            if len(full_pool) == 0:
                return qc

            gate = full_pool[-1]
            if gate['num_op'] > len(remaining_qubits):
                break
            else:
                gate = full_pool.pop()

            operands = remaining_qubits[:gate['num_op']]
            remaining_qubits = [q for q in remaining_qubits if q not in operands]

            num_params = gate['num_params']
            thetas_length += num_params
            thetas.resize(thetas_length)
            angles = thetas[thetas_length - num_params:thetas_length]
            op = gate['operation'](*angles)
            qc.append(op, operands)

    return qc




def by_num_cnot_old(metadata: MetadataSynthesis) -> qiskit.QuantumCircuit:
    pool = constant.operations_only_cnot
    # H,S,CX,RX,RY,RZ
    num_qubits = metadata.num_qubits
    depth = metadata.depth
    num_cnot = metadata.num_cnot
    num_gate = depth * num_qubits
    num_othergate = num_gate - num_cnot
    smallest_num_gate = 1 if int(0.15 * num_othergate) == 0 else int(0.15 * num_othergate)
    # Generate random num_gates for first 4 gates
    num_gates = generate_random_array(num_othergate, smallest_num_gate, 4)
    
    
    num_gates.append(num_cnot)

    qc = qiskit.QuantumCircuit(num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta')
    thetas_length = 0
    full_pool = []
    if len(pool) != len(num_gates):
        raise ValueError("The number of gates and the number of gates to be generated must be the same")
    for j, gate in enumerate(pool):
        full_pool.extend([gate]*num_gates[j])
    random.shuffle(full_pool)
    while(len(full_pool) > 0):
        remaining_qubits = list(range(num_qubits))
        random.shuffle(remaining_qubits)
        while(len(remaining_qubits) > 1):
            if len(full_pool) == 0:
                return qc
            gate = full_pool[-1]
            if gate['num_op'] > len(remaining_qubits):
                break
            else:
                gate = full_pool.pop()
            operands = remaining_qubits[:gate['num_op']]
            remaining_qubits = [q for q in remaining_qubits if q not in operands]

            num_params = gate['num_params']
            thetas_length += num_params
            thetas.resize(thetas_length)
            angles = thetas[thetas_length - num_params:thetas_length]
            op = gate['operation'](*angles)
            qc.append(op, operands)
    return qc
    
    
    

def by_num_cnot_old(metadata: Metadata) -> qiskit.QuantumCircuit:
    """
    Tạo mạch lượng tử với tổng các cổng không phải CX bằng metadata.num_qubits.
    """
    pool = constant["operations_only_cnot"]
    num_qubits = metadata.num_qubits
    num_cnot = metadata.num_cnot  # Số lượng cổng CNOT mong muốn
    total_gates = num_qubits  # Tổng số cổng không phải CX

    # Tính số lượng cổng khác CNOT (CX)
    num_othergate = total_gates
    if num_cnot > total_gates:
        raise ValueError("Số lượng num_cnot lớn hơn tổng số cổng num_qubits!")

    # Phân bổ số lượng cổng khác CX
    num_gates = generate_random_array(num_othergate, 1, len(pool) - 1)

    # Tạo danh sách cổng đầy đủ
    full_pool = []
    for j, gate in enumerate(pool):
        # Chỉ thêm cổng không phải CX
        if gate["name"] != "cx":
            full_pool.extend([gate] * num_gates[j])

    # Thêm cổng CX
    full_pool.extend([gate for gate in pool if gate["name"] == "cx"] * num_cnot)
    random.shuffle(full_pool)  # Trộn danh sách cổng ngẫu nhiên

    # Tạo mạch lượng tử
    qc = qiskit.QuantumCircuit(num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta')
    thetas_length = 0

    # Thêm cổng vào mạch
    while len(full_pool) > 0:
        remaining_qubits = list(range(num_qubits))
        random.shuffle(remaining_qubits)  # Trộn các qubit ngẫu nhiên
        while len(remaining_qubits) > 0:
            if len(full_pool) == 0:
                break
            gate = full_pool.pop()
            if gate["num_op"] > len(remaining_qubits):
                break
            operands = remaining_qubits[:gate["num_op"]]
            remaining_qubits = [q for q in remaining_qubits if q not in operands]

            # Thêm tham số động (nếu có)
            num_params = gate["num_params"]
            thetas_length += num_params
            thetas.resize(thetas_length)
            angles = thetas[thetas_length - num_params:thetas_length]
            op = gate["operation"](*angles)
            qc.append(op, operands)

    return qc



def by_num_qubits(metadata: MetadataSynthesis) -> qiskit.QuantumCircuit:
    """
    Generate a quantum circuit with a specified number of qubits and ensure len(qc.parameters) == num_qubits.

    Args:
        metadata (MetadataSynthesis): Contains the properties for the quantum circuit.

    Returns:
        qiskit.QuantumCircuit: Generated quantum circuit.
    """
    pool = constant.operations
    num_qubits = metadata.num_qubits
    depth = metadata.depth
    seed = None
    max_operands = 2

    while True:  # Repeat until a valid circuit is generated
        qc = qiskit.QuantumCircuit(num_qubits)
        rng = random.Random(seed)
        thetas = qiskit.circuit.ParameterVector('theta')
        thetas_length = 0

        for _ in range(depth):
            remaining_qubits = list(range(num_qubits))
            while remaining_qubits:
                max_possible_operands = min(len(remaining_qubits), max_operands)
                num_operands = rng.choice([1, 2]) if max_possible_operands >= 2 else 1
                rng.shuffle(remaining_qubits)
                operands = remaining_qubits[:num_operands]
                remaining_qubits = [q for q in remaining_qubits if q not in operands]

                num_op_pool = [item for item in pool if item['num_op'] == num_operands]
                operation = rng.choice(num_op_pool)
                num_params = operation['num_params']

                if num_params > 0:
                    thetas_length += num_params
                    thetas.resize(thetas_length)
                    angles = thetas[thetas_length - num_params:thetas_length]
                    op = operation['operation'](*angles)
                else:
                    op = operation['operation']()

                register_operands = [qc.qubits[i] for i in operands]
                qc.append(op, register_operands)

        # Ensure len(qc.parameters) == num_qubits
        if len(qc.parameters) == num_qubits:
            break

    return qc
