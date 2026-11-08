"""Generate an own-copy instrumented kernel with unchanged random draw order."""
import inspect
from . import ROOT,WORKSPACE

def build(overwrite=False):
    import forecast_rebuild  # Registers the existing project packages.
    from cooperative.packets import packet_interval
    source=inspect.getsource(packet_interval.py_func)
    def replace(old,new,count=1):
        nonlocal source
        if source.count(old)!=count:raise ValueError('Upstream instrumentation anchor changed: '+old[:70])
        source=source.replace(old,new)
    replace('def packet_interval(','def traced_packet_interval(')
    replace('    reserved_slots,\n):','    reserved_slots,\n    entry_time, first_tx, queue_wait, service_wait, queue_before, service_before, end_time, evidence,\n    no_jam, blocked, last_next, last_next_time, frame_extra,\n):')
    replace('    attempt_out.fill(0)','    frame_extra.fill(0)\n    attempt_out.fill(0)')
    replace('            born[pid] = next_birth','            born[pid] = next_birth\n            entry_time[pid] = next_birth\n            first_tx[pid] = -1.')
    replace('            if lengths[s] >= 64:\n                status[pid] = 4','            if lengths[s] >= 64:\n                status[pid] = 4\n                end_time[pid] = next_birth')
    replace('            status[pid] = 2\n            expire += 1','            status[pid] = 2\n            end_time[pid] = now\n            if first_tx[pid] >= 0:\n                service_wait[pid] += now-first_tx[pid]\n                service_before[pid] += max(0.,min(now,born[pid]+.25)-first_tx[pid])\n            else:\n                queue_wait[pid] += now-entry_time[pid]\n                queue_before[pid] += max(0.,min(now,born[pid]+.25)-entry_time[pid])\n            first_tx[pid] = -1.\n            expire += 1')
    replace('        if rx < 0:\n            continue','        if rx < 0:\n            if now <= born[pid]+.25+1e-10: evidence[pid,0] += 1\n            frame_extra[tx,0] += 1\n            continue')
    replace('        attempts += 1','        if first_tx[pid] < 0:\n            queue_wait[pid] += max(0.,now-entry_time[pid])\n            queue_before[pid] += max(0.,min(now,born[pid]+.25)-entry_time[pid])\n            first_tx[pid] = now\n        d = dest[pid]\n        if last_next[tx,d] >= 0 and last_next[tx,d] != rx: frame_extra[tx,1] += 1\n        last_next[tx,d] = rx\n        last_next_time[tx,d] = now\n        if tries[pid] > 0: frame_extra[tx,2] += 1\n        attempts += 1')
    replace('        if np.random.random() < current[tx, rx]:','        draw = np.random.random()\n        if draw >= current[tx,rx] and now <= born[pid]+.25+1e-10:\n            evidence[pid,1] += 1\n            if draw < no_jam[tx,rx]: evidence[pid,2] += 1\n            if blocked[tx,rx]: evidence[pid,3] += 1\n            else: evidence[pid,4] += 1\n        if draw < current[tx, rx]:')
    replace('            successful += 1','            service_wait[pid] += now+.005-first_tx[pid]\n            service_before[pid] += max(0.,min(now+.005,born[pid]+.25)-first_tx[pid])\n            first_tx[pid] = -1.\n            frame_extra[tx,3] += 1\n            successful += 1')
    replace('                status[pid] = 1\n                delivered += 1','                status[pid] = 1\n                end_time[pid] = now+.005\n                delivered += 1')
    replace('            elif lengths[rx] >= 64:\n                status[pid] = 4','            elif lengths[rx] >= 64:\n                status[pid] = 4\n                end_time[pid] = now+.005')
    replace('                queue[rx, pos] = pid','                queue[rx, pos] = pid\n                entry_time[pid] = now+.005')
    replace('            status[pid] = 3\n            retrydrop += 1','            status[pid] = 3\n            end_time[pid] = now+.005\n            service_wait[pid] += now+.005-first_tx[pid]\n            service_before[pid] += max(0.,min(now+.005,born[pid]+.25)-first_tx[pid])\n            first_tx[pid] = -1.\n            retrydrop += 1')
    header='"""Own instrumented copy, generated from frozen cooperative.packets; one RNG draw per attempted hop."""\nimport numpy as np\nfrom numba import njit\nfrom swarm_sim.fast_network import graph_statistics,pop_front\n\n'
    target=ROOT/'instrumented_packet.py';value=header+source
    if target.exists() and target.read_text(encoding='utf-8')!=value and not overwrite:raise ValueError('Generated instrumentation changed')
    target.write_text(value,encoding='utf-8')

if __name__=='__main__':build()
