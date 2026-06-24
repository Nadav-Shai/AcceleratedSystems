import matplotlib.pyplot as plt

# ── Paste your measurements here ──────────────────────────────────────────────
# (throughput in req/sec, median latency in ms)
data = [
    # (throughput, median_latency)
    (4668,  0.0408),   # load = maxLoad/10
    (13537,  0.0417),
    (22406, 0.0580),
    (31275, 0.4485),
    (40144, 0.5385),
    (49013, 16.2869),
    (57882, 29.1173),
    (66751, 39.3828),
    (75620, 48.662),
    (84489, 52.3978),   # load = maxLoad  (replace with your actual measured throughput)
    (93358, 52.3182),   # load = ~2*maxLoad
]
# ──────────────────────────────────────────────────────────────────────────────

throughputs = [d[0] for d in data]
latencies   = [d[1] for d in data]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(throughputs, latencies, marker='o', linewidth=2, markersize=6,
        color='steelblue', label='Streams server')

ax.set_xlabel('Throughput (req/sec)', fontsize=13)
ax.set_ylabel('Median Latency (ms)', fontsize=13)
ax.set_title('Latency vs. Throughput — CUDA Streams Server', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)

# Linear scale on X as required by the spec
ax.set_xscale('linear')

plt.tight_layout()
plt.savefig('latency_throughput_streams.png', dpi=150)
print("Saved to latency_throughput_streams.png")