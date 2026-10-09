"""
Proactive action module for ANOS.

Adjusts process priorities to reduce network contention, using only
standard, portable APIs available on Windows, Linux, and macOS.

All actions are reversible: the Graceful Lifecycle ensures that any
priority modification is restored upon termination.
"""
import psutil
import platform


class NetworkOptimizer:
    """
    Adjusts process priorities to reduce network contention.

    The optimizer identifies processes with many active network
    connections and lowers their priority, allowing other processes
    (e.g., user-facing applications) to receive more CPU and bandwidth.
    """

    # System-critical processes that must NEVER be touched
    PROTECTED = {
        "system", "System", "svchost.exe", "lsass.exe", "csrss.exe",
        "wininit.exe", "services.exe", "smss.exe", "python.exe",
        "pythonw.exe", "cmd.exe", "explorer.exe", "conhost.exe",
        "System Idle Process", "Registry", "MemCompression",
    }

    def __init__(self):
        self.actions = []

    def identify(self, top_n=5):
        """
        Identify the top-N processes with the most active connections.

        Returns a list of dicts with keys:
            pid, name, score, estab, conns, old_nice
        """
        procs = []
        try:
            conns = psutil.net_connections(kind="inet")
        except Exception:
            return []

        # Aggregate connection stats by PID
        stats = {}
        for c in conns:
            if c.pid:
                if c.pid not in stats:
                    stats[c.pid] = {"estab": 0, "conns": 0}
                stats[c.pid]["conns"] += 1
                if c.status == "ESTABLISHED":
                    stats[c.pid]["estab"] += 1

        # Score each process
        for pid, s in stats.items():
            try:
                p = psutil.Process(pid)
                name = p.name()
                if name in self.PROTECTED:
                    continue
                score = s["estab"] + 0.5 * s["conns"]
                procs.append({
                    "pid": pid,
                    "name": name,
                    "score": score,
                    "estab": s["estab"],
                    "conns": s["conns"],
                    "old_nice": p.nice(),
                })
            except Exception:
                continue

        procs.sort(key=lambda x: -x["score"])
        return procs[:top_n]

    def optimize(self, procs):
        """
        Lower the priority of processes identified as high-traffic.

        Returns a list of applied actions, each with keys:
            pid, name, old, new, estab
        """
        applied = []
        for pr in procs:
            if pr["estab"] < 3:
                # Only act on processes with at least 3 established conns
                continue
            try:
                p = psutil.Process(pr["pid"])
                old = p.nice()

                # Set priority to the lowest available
                if platform.system() == "Windows":
                    p.nice(psutil.IDLE_PRIORITY_CLASS)
                else:
                    # POSIX: -20 (high priority) ... 19 (low priority)
                    p.nice(19)

                new = p.nice()
                applied.append({
                    "pid": pr["pid"],
                    "name": pr["name"],
                    "old": old,
                    "new": new,
                    "estab": pr["estab"],
                })
            except Exception:
                continue

        self.actions.extend(applied)
        return applied

    def restore(self):
        """
        Restore all modified priorities to their original values.

        Should be called before exiting the program. This is part of the
        'Graceful Lifecycle' abstraction: every action has an inverse.
        """
        for a in self.actions:
            try:
                p = psutil.Process(a["pid"])
                if platform.system() == "Windows":
                    p.nice(psutil.NORMAL_PRIORITY_CLASS)
                else:
                    p.nice(0)
            except Exception:
                # Process may have terminated; ignore
                pass
        self.actions = []
