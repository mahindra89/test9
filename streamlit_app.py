import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import pandas as pd
import random

# --- Page Setup ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("STRF Scheduler")

# --- Algorithm Selection ---
algo = st.radio("Choose STRF Algorithm", ("STRF Scheduling with Quantum Time", "STRF Scheduling Without Quantum Time"))
st.markdown("---")

# --------------------------------------------
# ALGORITHM 1: STRF WITH QUANTUM TIME 
# --------------------------------------------
if algo == "STRF Scheduling with Quantum Time":
    st.title("STRF Scheduling with Quantum Time")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_jobs = st.number_input("Number of Jobs", min_value=1, max_value=10, value=4)
    with col2:
        num_cpus = st.number_input("Number of CPUs", min_value=1, max_value=4, value=2)
    with col3:
        chunk_unit = st.number_input("Chunk Unit (e.g., 0.5, 1.0):", value=1.0)
    with col4:
        quantum_time = st.number_input("Quantum Time", value=2.0)

    def get_random_half_step(min_val, max_val):
        steps = int((max_val - min_val) * 2) + 1
        return round(min_val + 0.5 * random.randint(0, steps - 1), 1)

    if 'random_values' not in st.session_state:
        st.session_state.random_values = []

    if st.button("Randomize Job Times"):
        st.session_state.random_values = [
            {'arrival': get_random_half_step(0, 5), 'burst': get_random_half_step(1, 10)}
            for _ in range(num_jobs)
        ]

    processes = []
    for i in range(num_jobs):
        st.markdown(f"### Job J{i+1}")
        default_arrival = st.session_state.random_values[i]['arrival'] if i < len(st.session_state.random_values) else 0.0
        default_burst = st.session_state.random_values[i]['burst'] if i < len(st.session_state.random_values) else 3.0
        c1, c2 = st.columns(2)
        with c1:
            arrival = st.number_input(f"Arrival Time for J{i+1}", value=default_arrival, key=f"arrival_{i}")
        with c2:
            burst = st.number_input(f"Burst Time for J{i+1}", value=default_burst, key=f"burst_{i}")
        processes.append({'id': f'J{i+1}', 'arrival_time': arrival, 'burst_time': burst})

    if st.button("Run Simulation"):
        from strf_quantum import run_quantum_simulation
        run_quantum_simulation(processes, num_cpus, chunk_unit, quantum_time)

# --------------------------------------------
# ALGORITHM 2: STRF WITHOUT QUANTUM TIME
# --------------------------------------------
elif algo == "STRF Scheduling Without Quantum Time":
    st.title("STRF Scheduling Without Quantum Time")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_jobs = st.number_input("Number of Jobs", 1, 10, 3, key="num_jobs_woq")
    with col2:
        num_cpus = st.number_input("Number of CPUs", 1, 4, 2, key="num_cpus_woq")
    with col3:
        chunk_unit = st.number_input("Chunk Unit (e.g., 0.5, 1.0):", value=1.0, key="chunk_unit_woq")

    if st.button("Randomize Job Times", key="randomize_woq"):
        st.session_state.special_jobs = [
            {"arrival": round(random.uniform(0, 5) * 2) / 2, "burst": round(random.uniform(1, 10) * 2) / 2}
            for _ in range(num_jobs)
        ]

    processes = []
    for i in range(num_jobs):
        st.subheader(f"Job J{i+1}")
        default_arrival = st.session_state.get("special_jobs", [{}]*num_jobs)[i].get("arrival", 0.0)
        default_burst = st.session_state.get("special_jobs", [{}]*num_jobs)[i].get("burst", 3.0)
        c1, c2 = st.columns(2)
        with c1:
            arrival = st.number_input(f"Arrival Time for J{i+1}", value=default_arrival, key=f"a_{i}")
        with c2:
            burst = st.number_input(f"Burst Time for J{i+1}", value=default_burst, key=f"b_{i}")
        processes.append({'id': f'J{i+1}', 'arrival_time': arrival, 'burst_time': burst})

    if st.button("Run Special STRF"):
        from strf_without_quantum import run_without_quantum_simulation
        run_without_quantum_simulation(processes, num_cpus, chunk_unit)
