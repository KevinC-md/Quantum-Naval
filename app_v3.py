from flask import Flask, render_template_string, request, redirect, url_for
import numpy as np
import random, re, math
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import *
from qiskit_aer import AerSimulator

# ======================
# Choose a fake backend
# ======================
# Get all fake backend class names dynamically (only those that start with "Fake")
available_backend_classes = [name for name in globals() if name.startswith("Fake")]
valid_backends = []
for backend_name in available_backend_classes:
    try:
        backend = eval(backend_name)()  # Instantiate backend
        if backend.configuration().n_qubits >= 4:
            valid_backends.append(backend)
    except Exception:
        pass  # Ignore backends that cannot be instantiated

if not valid_backends:
    raise ValueError("No available fake backends with at least 4 qubits.")
device_backend = random.choice(valid_backends)
print(f"Selected backend: {device_backend.name()}")

# ==========================
# Flask and Global Game State
# ==========================
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

# We use a global dictionary to hold the game state (for a simple demo)
GAME_STATE = {}

def init_game():
    n = 4
    # Create two boards (tableros) as a list-of-lists with numbers 1..16
    tablero1 = []
    tablero2 = []
    k = 1
    for i in range(n):
        row1 = []
        row2 = []
        for j in range(n):
            row1.append(str(k))
            row2.append(str(k))
            k += 1
        tablero1.append(row1)
        tablero2.append(row2)
    
    GAME_STATE.clear()
    GAME_STATE.update({
        "n": n,
        "tablero1": tablero1,
        "tablero2": tablero2,
        "barcos1": ["" for _ in range(n)],  # ships for player 1
        "barcos2": ["" for _ in range(n)],  # ships for player 2
        "hits1": ["" for _ in range(100)],
        "hits2": ["" for _ in range(100)],
        "j1": n,  # remaining ships for player 1
        "j2": n,  # remaining ships for player 2
        "turn": 0,
        "Cj1": True,  # flag for Copenhagen move (each player only once)
        "Cj2": True,
        "current_player": 1,  # whose turn it is (1 or 2)
        "log": []  # game messages
    })

def log_message(msg):
    GAME_STATE["log"].append(msg)

# A helper to convert a board (list-of-lists) into an HTML table
def render_board(tablero):
    html = "<table border='1' style='text-align: center; border-collapse: collapse;'>"
    for row in tablero:
        html += "<tr>"
        for cell in row:
            html += f"<td style='padding: 10px;'>{cell}</td>"
        html += "</tr>"
    html += "</table>"
    return html

# ----------------------
# Game Logic Functions
# ----------------------
# (Most functions below are adapted from your original code.
# They now update the GAME_STATE and log messages instead of printing or using input.)

def set_ships(player, ships):
    """Assign the list of ship positions for the given player."""
    if player == 1:
        GAME_STATE["barcos1"] = ships
    else:
        GAME_STATE["barcos2"] = ships
    log_message(f"Player {player} ships set to: {ships}")

def convert_to_coord(number, matrix_size=4):
    """Convert a board number (1..16) into (row, col)."""
    if 1 <= number <= matrix_size**2:
        row = (number - 1) // matrix_size
        col = (number - 1) % matrix_size
        return (row, col)
    return (None, None)

def ataque(t, barcos, tablero, jugadorA, hits):
    bol_at = True  # do not allow attacking the same target repeatedly
    if t in hits:
        bol_at = False
    # Record the hit in the appropriate hits array:
    if jugadorA == 1:
        GAME_STATE["hits1"][GAME_STATE["turn"]] = t
    else:
        GAME_STATE["hits2"][GAME_STATE["turn"]] = t
    if (t in barcos) and bol_at:
        comparar(t, tablero, "🟢", "💥 HIT 💥")
        if jugadorA == 1:
            GAME_STATE["j2"] -= 1  # reduce opponent’s ships
        else:
            GAME_STATE["j1"] -= 1
    else:
        comparar(t, tablero, "🔴", "🕳️ FAIL 🕳️")

def comparar(t, tablero, accion, output):
    # Mark the board (tablero) where cell equals t with the given action symbol.
    x, y = -1, -1
    n = 4
    for i in range(n):
        for j in range(n):
            if t == tablero[i][j]:
                x, y = i, j
                break
    if x != -1 and y != -1:
        tablero[x][y] = accion
    log_message(output)

def coordenadas(i, j):
    """For the Copenhagen move: convert cell coordinates to a number using a quantum circuit."""
    bin_index = format(4 * i + j, '04b')
    qc = QuantumCircuit(4, 4)
    for qubit in range(4):
        if bin_index[3 - qubit] == '1':  # note the reversed order
            qc.x(qubit)
    qc.measure(range(4), range(4))
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1024)
    result = job.result()
    counts = result.get_counts()
    # Return the number corresponding to the first measurement outcome.
    for state in counts:
        return int(state, 2) + 1

def tuneling_attack(casilla_robo, tableroB, barcosB):
    """Quantum tunneling attack using a 3-qubit circuit."""
    circ = QuantumCircuit(3)
    # Check if the board cell contains a ship.
    if tableroB[casilla_robo[0]][casilla_robo[1]] in barcosB:
        log_message("🟣🔱 BARCO TUNELADO 🔱🟣")
        circ.x(0)
        circ.h(1)
        circ.cx(1, 2)
        circ.cx(0, 1)
        circ.h(0)
        simulator = AerSimulator(method='statevector')
        job = simulator.run(circ)
        res = job.result()
        state = res.get_statevector()
        return state, tableroB
    else:
        log_message("🟣 FAIL CUÁNTICO 🟣")
        return None, tableroB

def copenhague_attack(tablero_oponente):
    """Copenhagen attack using an oracle and diffusion operators."""
    num_q = len(sum(tablero_oponente, []))  # total number of cells
    circ = QuantumCircuit(num_q)
    for i in range(num_q):
        circ.h(i)
    # Apply an oracle: for every cell that contains a ship (here marked as 'barco'),
    # flip the qubit.
    for i in range(num_q):
        if tablero_oponente[i // 4][i % 4] == 'barco':
            circ.x(i)
    circ.h(num_q - 1)
    circ.mcx(list(range(num_q - 1)), num_q - 1)
    circ.h(num_q - 1)
    for i in range(num_q):
        if tablero_oponente[i // 4][i % 4] == 'barco':
            circ.x(i)
    # Diffusion operator:
    for qubit in range(num_q):
        circ.h(qubit)
    for qubit in range(num_q):
        circ.x(qubit)
    circ.h(num_q - 1)
    circ.mcx(list(range(num_q - 1)), num_q - 1)
    circ.h(num_q - 1)
    for qubit in range(num_q):
        circ.x(qubit)
    for qubit in range(num_q):
        circ.h(qubit)
    simulator = AerSimulator(method='statevector')
    job = simulator.run(circ)
    res = job.result()
    estado = np.array(res.get_statevector())
    probabilidades = [abs(amplitude)**2 for amplitude in estado]
    casillas_ordenadas = sorted(range(num_q), key=lambda i: probabilidades[i], reverse=True)
    estimacion = set()
    while len(estimacion) < 3:
        estimacion.add(random.choice(casillas_ordenadas[:num_q // 2]))
    estimacion_coordenadas = [(i // 4, i % 4) for i in estimacion]
    return estimacion_coordenadas

def bin2dec(bin_):
    decimal = 0
    for digit in bin_:
        decimal = decimal * 2 + int(digit)
    return decimal

def Qtarget_n(n):
    """Quantum target move generating a list of possible cell numbers."""
    simulator = AerSimulator()
    circ = QuantumCircuit(n, n)
    circ.h(0)
    for idx in range(n - 1):
        circ.cx(idx, idx + 1)
    circ.measure_all()
    tcirc = transpile(circ, simulator)
    job = simulator.run(tcirc, shots=1024)
    result = job.result()
    counts_noise = result.get_counts(0)
    lcounts = list(counts_noise.keys())
    for i in range(len(lcounts)):
        lcounts[i] = str(bin2dec(lcounts[i].replace(" 0000", "")) + 1)
    return lcounts

def Qtarget_():
    """Quantum target move returning 1 or 0 based on a GHZ-like circuit."""
    circ_ghz = QuantumCircuit(3, 3)
    circ_ghz.h(0)
    circ_ghz.cx(0, 1)
    circ_ghz.cx(0, 2)
    circ_ghz.barrier()
    circ_ghz.measure([0, 1, 2], [0, 1, 2])
    simulator = AerSimulator()
    job = simulator.run(circ_ghz, shots=1)
    result = job.result()
    counts = result.get_counts(0)
    result_str = list(counts.keys())[0]
    if '1' in result_str:
        return 1
    else:
        return 0

def process_move(player, move):
    """
    Process a move string for the current player.
    Depending on the input (e.g., containing "Q", "T:", or "C") different move types are executed.
    """
    opponent = 2 if player == 1 else 1
    # Use opponent’s board and ship list.
    if opponent == 1:
        barcos = GAME_STATE["barcos1"]
        tablero = GAME_STATE["tablero1"]
        hits = GAME_STATE["hits1"]
    else:
        barcos = GAME_STATE["barcos2"]
        tablero = GAME_STATE["tablero2"]
        hits = GAME_STATE["hits2"]
    
    t = move.strip()
    
    # Quantum move
    if "Q" in t:
        t = re.sub(r'[a-zA-Z]', '', t)
        log_message("🔱 Quantum move initiated!")
        QQ = Qtarget_()
        log_message(f"Quantum move result: {QQ}")
        if QQ:
            ataque(t, barcos, tablero, player, hits)
            log_message("🟣🔱 ATAQUE CUÁNTICO 🔱🟣")
            hitsnew = np.setdiff1d(hits, [''])
            try:
                hits_int = [int(x) for x in hitsnew if x != ""]
            except Exception:
                hits_int = []
            Qtarget = Qtarget_n(4)
            for h in hits_int:
                if str(h) in Qtarget:
                    Qtarget.remove(str(h))
            t_ = random.choice(Qtarget) if Qtarget else t
            GAME_STATE["turn"] += 1
            if player == 1:
                GAME_STATE["hits1"][GAME_STATE["turn"]] = t_
            else:
                GAME_STATE["hits2"][GAME_STATE["turn"]] = t_
            ataque(t_, barcos, tablero, player, hits)
        else:
            log_message("🟣 FAIL CUÁNTICO 🟣")
    
    # Tunneling move: expected format "T:barcoB,posA"
    elif "T:" in t:
        parts = t.split(':')
        if len(parts) > 1:
            values = parts[1].split(',')
            if len(values) >= 2:
                barcoB = values[0].strip()
                # posA is not further used
                log_message("🔱 TUNELAMIENTO move initiated!")
                Tun_res, _ = tuneling_attack(convert_to_coord(int(barcoB)), tablero, barcos)
                ataque(barcoB, barcos, tablero, player, hits)
                if barcoB in barcos:
                    barcos.remove(barcoB)
                if player == 2:
                    if Tun_res is not None:
                        GAME_STATE["j2"] += 1
                        GAME_STATE["barcos2"].append(barcoB)
                else:
                    if Tun_res is not None:
                        GAME_STATE["j1"] += 1
                        GAME_STATE["barcos1"].append(barcoB)
            else:
                log_message("Invalid tunneling move format!")
    
    # Copenhagen move
    elif "C" in t:
        if (player == 1 and GAME_STATE["Cj1"]) or (player == 2 and GAME_STATE["Cj2"]):
            log_message("🔱 COPENHAGEN move initiated!")
            if player == 2:
                L_watch = copenhague_attack(GAME_STATE["tablero1"])
                barcos_op = GAME_STATE["barcos1"]
            else:
                L_watch = copenhague_attack(GAME_STATE["tablero2"])
                barcos_op = GAME_STATE["barcos2"]
            r_watch = []
            for tup in L_watch:
                r = coordenadas(tup[0], tup[1])
                r_watch.append(str(r))
            common_ = [num for num in r_watch if num in barcos_op]
            for i in common_:
                comparar(i, tablero, "🔵", "👀 BARCO 🛳️")
            if common_:
                if player == 2:
                    GAME_STATE["Cj2"] = False
                else:
                    GAME_STATE["Cj1"] = False
            else:
                log_message("🟣 FAIL CUÁNTICO 🟣")
        else:
            log_message("COPENHAGEN move can only be used once!")
    
    # Normal move
    else:
        ataque(t, barcos, tablero, player, hits)
    
    log_message(f"--- Board for Player {opponent} ---")
    board_str = render_board(tablero)
    log_message(board_str)

# =====================
# Flask Routes (Views)
# =====================

@app.route("/")
def index():
    init_game()  # reset game state on new session
    return render_template_string("""
    <h1>🔱 QuantumTarget 🔱</h1>
    <p>Welcome to Quantum Battleship!</p>
    <p><a href="{{ url_for('setup') }}">Start New Game</a></p>
    """)

@app.route("/setup", methods=["GET", "POST"])
def setup():
    if request.method == "POST":
        # Retrieve ship positions from the form for both players.
        barcos1 = [request.form.get(f"p1_ship_{i}") for i in range(1, 5)]
        barcos2 = [request.form.get(f"p2_ship_{i}") for i in range(1, 5)]
        set_ships(1, barcos1)
        set_ships(2, barcos2)
        log_message("Both players have placed their ships.")
        return redirect(url_for("game"))
    return render_template_string("""
    <h2>Ship Placement</h2>
    <form method="post">
      <h3>Player 1: Enter positions for your 4 ships (enter a number 1–16)</h3>
      {% for i in range(1,5) %}
         <label>Ship {{ i }}: <input type="text" name="p1_ship_{{ i }}"></label><br>
      {% endfor %}
      <h3>Player 2: Enter positions for your 4 ships (enter a number 1–16)</h3>
      {% for i in range(1,5) %}
         <label>Ship {{ i }}: <input type="text" name="p2_ship_{{ i }}"></label><br>
      {% endfor %}
      <input type="submit" value="Submit Ships">
    </form>
    """)

@app.route("/game", methods=["GET", "POST"])
def game():
    if request.method == "POST":
        move = request.form.get("move")
        current_player = GAME_STATE.get("current_player", 1)
        process_move(current_player, move)
        # Check if either player has lost all ships.
        if GAME_STATE["j1"] == 0 or GAME_STATE["j2"] == 0:
            return redirect(url_for("game_over"))
        # Switch turn: if current player was 1, now 2 and vice versa.
        GAME_STATE["current_player"] = 2 if current_player == 1 else 1
        return redirect(url_for("game"))
    
    tablero1_html = render_board(GAME_STATE["tablero1"])
    tablero2_html = render_board(GAME_STATE["tablero2"])
    log_html = "<br>".join(GAME_STATE["log"])
    return render_template_string("""
    <h1>Quantum Battleship</h1>
    <p>Current turn: Player {{ current_player }}</p>
    <h2>Player 1 Board</h2>
    {{ tablero1|safe }}
    <h2>Player 2 Board</h2>
    {{ tablero2|safe }}
    <h3>Game Log</h3>
    <div style="background-color: #f0f0f0; padding: 10px;">{{ log|safe }}</div>
    <form method="post">
      <label>Enter your move (e.g., a board number, "Q" for Quantum, "T:..." for Tunneling, "C" for Copenhagen):</label><br>
      <input type="text" name="move">
      <input type="submit" value="Submit Move">
    </form>
    """, current_player=GAME_STATE["current_player"],
       tablero1=tablero1_html, tablero2=tablero2_html, log=log_html)

@app.route("/game_over")
def game_over():
    winner = "Player 1" if GAME_STATE["j2"] == 0 else "Player 2"
    return render_template_string("""
    <h1>Game Over!</h1>
    <p>{{ winner }} wins!</p>
    <p><a href="{{ url_for('index') }}">Start a New Game</a></p>
    """, winner=winner)

# =====================
# Run the Flask App
# =====================
if __name__ == '__main__':
    app.run(debug=True)
