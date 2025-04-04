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
algo = st.radio("Choose STRF Algorithm", (
    "STRF Scheduling with Quantum Time",
    "STRF Scheduling Without Quantum Time"
))
st.markdown("---")

# --- Utility Function for Drawing Gantt Chart ---
def draw_gantt_chart(gantt_data, queue_snapshots, end_time, cpu_names, processes, quantum_time=None):
    max_time = max(end_time.values())
    fig, ax = plt.subplots(figsize=(18, 8))
    cmap = plt.colormaps.get_cmap('tab20')
    colors = {p['id']: mcolors.to_hex(cmap(i / max(len(processes), 1))) for i, p in enumerate(processes)}
    y_pos = {cpu: len(cpu_names) - idx for idx, cpu in enumerate(cpu_names)}

    for start, cpu, job, duration in gantt_data:
        y = y_pos[cpu]
        ax.barh(y, duration, left=start, color=colors[job], edgecolor='black')
        ax.text(start + duration / 2, y, job, ha='center', va='center', color='white', fontsize=9)

    for t in range(int(max_time) + 1):
        if quantum_time and t % int(quantum_time) == 0:
            ax.axvline(x=t, color='red', linestyle='-', linewidth=0.5, alpha=0.6)
        else:
            ax.axvline(x=t, color='black', linestyle='--', alpha=0.2)

    for time, jobs in queue_snapshots:
        for i, (jid, rem) in enumerate(jobs):
            y = -1 - i * 0.6
            rect = patches.Rectangle((time - 0.25, y - 0.25), 0.5, 0.5, edgecolor='black', facecolor='white')
            ax.add_patch(rect)
            ax.text(time, y, f"{jid}={rem}", ha='center', va='center', fontsize=7)

    max_q = max((len(q[1]) for q in queue_snapshots), default=0)
    ax.set_ylim(-1 - max_q * 0.6 - 0.5, len(cpu_names) + 1)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(cpu_names)
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart")

    if quantum_time:
        ax.legend([Line2D([0], [0], color='red', lw=2)], ['Quantum Marker'], loc='upper right')

    plt.grid(axis='x')
    plt.tight_layout()
    return fig

# --- Function: STRF With Quantum Time ---
def run_with_quantum():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_jobs = st.number_input("Number of Jobs", 1, 10, 4, key="wq_jobs")
    with col2:
        num_cpus = st.number_input("Number of CPUs", 1, 4, 2, key="wq_cpus")
    with col3:
        chunk_unit = st.number_input("Chunk Unit (e.g., 0.5, 1.0)", value=1.0, key="wq_chunk")
    with col4:
        quantum_time = st.number_input("Quantum Time", value=2.0, key="wq_quantum")

    if st.button("Randomize Job Times", key="wq_rand"):
        st.session_state.random_jobs = [
            {'arrival': round(random.uniform(0, 5) * 2) / 2, 'burst': round(random.uniform(1, 10) * 2) / 2}
            for _ in range(num_jobs)
        ]

    processes = []
    for i in range(num_jobs):
        default = st.session_state.get("random_jobs", [{}]*num_jobs)[i]
        c1, c2 = st.columns(2)
        with c1:
            arrival = st.number_input(f"Arrival Time for J{i+1}", value=default.get('arrival', 0.0), key=f"wq_arr_{i}")
        with c2:
            burst = st.number_input(f"Burst Time for J{i+1}", value=default.get('burst', 3.0), key=f"wq_burst_{i}")
        processes.append({'id': f'J{i+1}', 'arrival_time': arrival, 'burst_time': burst})

    if st.button("Run Simulation", key="wq_run"):
        simulate(processes, num_cpus, chunk_unit, quantum_time)

# --- Function: STRF Without Quantum Time ---
def run_without_quantum():
    col1, col2, col3 = st.columns(3)
    with col1:
        num_jobs = st.number_input("Number of Jobs", 1, 10, 4, key="woq_jobs")
    with col2:
        num_cpus = st.number_input("Number of CPUs", 1, 4, 2, key="woq_cpus")
    with col3:
        chunk_unit = st.number_input("Chunk Unit (e.g., 0.5, 1.0)", value=1.0, key="woq_chunk")

    if st.button("Randomize Job Times", key="woq_rand"):
        st.session_state.special_jobs = [
            {'arrival': round(random.uniform(0, 5) * 2) / 2, 'burst': round(random.uniform(1, 10) * 2) / 2}
            for _ in range(num_jobs)
        ]

    processes = []
    for i in range(num_jobs):
        default = st.session_state.get("special_jobs", [{}]*num_jobs)[i]
        c1, c2 = st.columns(2)
        with c1:
            arrival = st.number_input(f"Arrival Time for J{i+1}", value=default.get('arrival', 0.0), key=f"woq_arr_{i}")
        with c2:
            burst = st.number_input(f"Burst Time for J{i+1}", value=default.get('burst', 3.0), key=f"woq_burst_{i}")
        processes.append({'id': f'J{i+1}', 'arrival_time': arrival, 'burst_time': burst})

    if st.button("Run Special STRF", key="woq_run"):
        simulate(processes, num_cpus, chunk_unit, quantum_time=None)

# --- Core Simulation Logic ---
def simulate(processes, num_cpus, chunk_unit, quantum_time=None):
    arrival_time = {p['id']: p['arrival_time'] for p in processes}
    burst_time = {p['id']: p['burst_time'] for p in processes}
    remaining_time = burst_time.copy()
    job_chunks, start_time, end_time = {}, {}, {}
    for job_id, bt in burst_time.items():
        chunks, rem = [], bt
        while rem > 0:
            chunk = min(chunk_unit, rem)
            chunks.append(chunk)
            rem -= chunk
        job_chunks[job_id] = chunks

    cpu_names = [f"CPU{i+1}" for i in range(num_cpus)]
    busy_until = {cpu: 0 for cpu in cpu_names}
    current_jobs = {cpu: None for cpu in cpu_names}
    busy_jobs = set()
    gantt_data, queue_snapshots = [], []
    current_time, jobs_completed = 0, 0
    next_sched = 0

    def capture_queue(time):
        queue = sorted([j for j in remaining_time if remaining_time[j] > 0 and arrival_time[j] <= time and j not in busy_jobs],
                       key=lambda j: (remaining_time[j], arrival_time[j]))
        if queue:
            queue_snapshots.append((time, [(j, round(remaining_time[j], 1)) for j in queue]))

    capture_queue(current_time)

    while jobs_completed < len(processes):
        for cpu in cpu_names:
            if busy_until[cpu] <= current_time and current_jobs[cpu]:
                busy_jobs.discard(current_jobs[cpu])
                current_jobs[cpu] = None

        available_cpus = [c for c in cpu_names if busy_until[c] <= current_time and current_jobs[c] is None]
        can_schedule = quantum_time is None or current_time >= next_sched

        if available_cpus and can_schedule:
            capture_queue(current_time)
            queue = sorted([j for j in remaining_time if remaining_time[j] > 0 and arrival_time[j] <= current_time and j not in busy_jobs],
                           key=lambda j: (remaining_time[j], arrival_time[j]))
            for cpu in available_cpus:
                if not queue: break
                job = queue.pop(0)
                chunk = job_chunks[job].pop(0)
                if job not in start_time:
                    start_time[job] = current_time
                current_jobs[cpu] = job
                busy_jobs.add(job)
                busy_until[cpu] = current_time + chunk
                remaining_time[job] -= chunk
                gantt_data.append((current_time, cpu, job, chunk))
                if remaining_time[job] < 1e-3:
                    end_time[job] = current_time + chunk
                    jobs_completed += 1
            if quantum_time:
                next_sched = current_time + quantum_time

        future_events = (
            [busy_until[c] for c in cpu_names if busy_until[c] > current_time] +
            [arrival_time[j] for j in arrival_time if arrival_time[j] > current_time and remaining_time[j] > 0]
        )
        if quantum_time and next_sched > current_time:
            future_events.append(next_sched)
        current_time = min(future_events) if future_events else current_time + 0.1

    for p in processes:
        p['start_time'] = start_time[p['id']]
        p['end_time'] = end_time[p['id']]
        p['turnaround_time'] = p['end_time'] - p['arrival_time']

    df = pd.DataFrame([{
        "Job": p['id'],
        "Arrival": p['arrival_time'],
        "Burst": p['burst_time'],
        "Start": round(p['start_time'], 1),
        "End": round(p['end_time'], 1),
        "Turnaround": round(p['turnaround_time'], 1)
    } for p in processes])
    avg_tat = sum(p['turnaround_time'] for p in processes) / len(processes)

    st.subheader("Result Table")
    st.dataframe(df, use_container_width=True)
    st.markdown(f"**Average Turnaround Time:** `{avg_tat:.2f}`")
    st.subheader("Gantt Chart")
    st.pyplot(draw_gantt_chart(gantt_data, queue_snapshots, end_time, cpu_names, processes, quantum_time), use_container_width=True)

# --- Run Based on Choice ---
if algo == "STRF Scheduling with Quantum Time":
    run_with_quantum()
elif algo == "STRF Scheduling Without Quantum Time":
    run_without_quantum()
