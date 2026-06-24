import matplotlib.pyplot as plt

# ── Paste your measurements here ──────────────────────────────────────────────
# Each entry is (throughput in req/sec, median latency in ms)

streams = [
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

queue_1024 = [
    (8080.0, 0.0432),
    (16150.3, 0.0434),
    (24256.0, 0.0424),
    (32294.1, 0.0442),
    (40439.5, 0.0464),
    (48576.2, 0.0497),
    (56493.4, 0.0499),
    (64685.3, 0.0559),
    (72761.0, 0.0525),
    (80810.1, 0.0563),
    (102807.8, 18.1115),
]

queue_512 = [
    (8194.1,  0.5447),
    (16402.6, 0.0654),
    (24596.0, 0.0771),
    (32785.4, 0.0719),
    (41018.0, 0.0767),
    (49279.9, 0.0682),
    (57457.4, 0.0714),
    (65686.0, 0.0805),
    (74052.8, 0.0917),
    (82090.5, 0.0891),
    (72932.9, 37.1501),  # 2*maxLoad
]

queue_256 = [
    (8597.9,   0.1020),
    (17174.7,  0.0944),
    (25834.9,  0.0954),
    (34348.5,  0.1044),
    (42782.2,  0.1413),
    (51449.8,  0.1476),
    (60118.3,  0.1261),
    (68953.0,  0.1772),
    (65509.7,  0.1688),
    (85971.9,  0.1373),
    (116183.9, 14.3750),  # 2*maxLoad
]
# ──────────────────────────────────────────────────────────────────────────────

def unzip(data):
    return [d[0] for d in data], [d[1] for d in data]

fig, ax = plt.subplots(figsize=(10, 6))

for data, label, marker in [
    (streams,    'Streams',         'o'),
    (queue_1024, 'Queue (1024 threads)', 's'),
    (queue_512,  'Queue (512 threads)',  '^'),
    (queue_256,  'Queue (256 threads)',  'D'),
]:
    throughputs, latencies = unzip(data)
    ax.plot(throughputs, latencies, marker=marker, linewidth=2,
            markersize=6, label=label)

ax.set_xlabel('Throughput (req/sec)', fontsize=13)
ax.set_ylabel('Median Latency (ms)', fontsize=13)
ax.set_title('Latency vs. Throughput — Streams vs. Queue Server', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xscale('linear')

plt.tight_layout()
plt.savefig('latency_throughput_combined.png', dpi=150)
print("Saved to latency_throughput_combined.png")