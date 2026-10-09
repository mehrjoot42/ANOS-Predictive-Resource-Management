"""
System monitoring module for ANOS.

Provides NetworkMonitor for reading real system telemetry through
standard, portable APIs (psutil).
"""
import time
import psutil
import numpy as np
from collections import deque, defaultdict


class NetworkMonitor:
    """Reads real system telemetry using only standard, portable APIs."""

    def __init__(self):
        self.last_net = None
        self.last_time = None
        self.speed_history = deque(maxlen=30)

    def read_counters(self):
        """Read raw network counters from the OS."""
        n = psutil.net_io_counters()
        return {
            "recv": n.bytes_recv,
            "sent": n.bytes_sent,
            "pkt_recv": n.packets_recv,
            "pkt_sent": n.packets_sent,
            "err": n.errin + n.errout + n.dropin + n.dropout,
        }

    def measure(self):
        """Compute instantaneous network metrics (speed, jitter, noise)."""
        now = time.time()
        net = self.read_counters()
        dl, ul = 0.0, 0.0
        if self.last_net and self.last_time:
            dt = now - self.last_time
            if dt > 0.01:
                dl = max(0.0, (net["recv"] - self.last_net["recv"]) / (1024**2) / dt)
                ul = max(0.0, (net["sent"] - self.last_net["sent"]) / (1024**2) / dt)
        self.last_net = net
        self.last_time = now
        self.speed_history.append(dl)

        jitter = float(np.std(self.speed_history)) if len(self.speed_history) > 3 else 0.0
        total_pkts = net["pkt_recv"] + net["pkt_sent"]
        pkt_loss = 100 * net["err"] / total_pkts if total_pkts > 0 else 0.0
        noise = jitter * 10 + pkt_loss

        return {
            "dl": dl,
            "ul": ul,
            "jitter": jitter,
            "pkt_loss": pkt_loss,
            "noise": noise,
        }

    def get_heavy_processes(self, top_n=5):
        """Identify processes with the most active network connections."""
        try:
            conns = psutil.net_connections(kind="inet")
        except (psutil.AccessDenied, PermissionError):
            return []
        
        stats = defaultdict(lambda: {"estab": 0, "conns": 0})
        for c in conns:
            if c.pid:
                stats[c.pid]["conns"] += 1
                if c.status == "ESTABLISHED":
                    stats[c.pid]["estab"] += 1
        
        result = []
        for pid, s in stats.items():
            try:
                p = psutil.Process(pid)
                result.append({
                    "pid": pid,
                    "name": p.name(),
                    "estab": s["estab"],
                    "conns": s["conns"],
                    "mem_mb": p.memory_info().rss / (1024**2),
                    "score": s["estab"] + 0.5 * s["conns"],
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        result.sort(key=lambda x: -x["score"])
        return result[:top_n]

    def snapshot(self):
        """Take a full snapshot of current system telemetry."""
        s = self.measure()
        s["cpu"] = psutil.cpu_percent(interval=None)
        s["ram"] = psutil.virtual_memory().percent
        s["swap"] = psutil.swap_memory().percent
        s["ts"] = time.time()
        return s
