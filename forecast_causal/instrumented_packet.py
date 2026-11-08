"""Own instrumented copy, generated from frozen cooperative.packets; one RNG draw per attempted hop."""
import numpy as np
from numba import njit
from swarm_sim.fast_network import graph_statistics,pop_front

@njit(cache=True)
def traced_packet_interval(
    p_t,
    node_ids_t,
    visible,
    ns,
    dt,
    t_index,
    disable_relays,
    known,
    observed,
    payload,
    source,
    dest,
    born,
    status,
    tries,
    hops,
    relay_hops,
    delays,
    last_received,
    cursor,
    queue,
    lengths,
    ic,
    fc,
    attempt_out,
    success_out,
    last_ack_out,
    queue_out,
    frame_out,
    reserved_slots,
    entry_time, first_tx, queue_wait, service_wait, queue_before, service_before, end_time, evidence,
    no_jam, blocked, last_next, last_next_time, frame_extra,
):
    n = p_t.shape[0]
    nc = visible.shape[1]
    slots = int(round(dt / 0.005))
    ring = payload.shape[0]
    frame_extra.fill(0)
    attempt_out.fill(0)
    success_out.fill(0)
    last_ack_out.fill(-1.0)
    generated = ic[0]
    delivered = ic[1]
    attempts = ic[2]
    successful = ic[3]
    retries = ic[4]
    expire = ic[5]
    retrydrop = ic[6]
    overflow = ic[7]
    timely = ic[8]
    relay_successes = ic[9]
    global_slot = ic[10]
    outevents = ic[11]
    next_birth = fc[0]
    aoisum = fc[1]
    outage = fc[2]
    outagemax = fc[3]
    outagelen = fc[4]
    t = t_index
    current = p_t.copy()
    if disable_relays:
        for i in range(ns, n):
            current[i, :] = 0
            current[:, i] = 0
    routing, stat, a = graph_statistics(current, ns)
    if stat[0] < 0.5:
        outage += dt
        outagelen += dt
        if outagelen == dt:
            outevents += 1
        outagemax = max(outagemax, outagelen)
    else:
        outagelen = 0
    now = t * dt
    if abs(now / 3 - round(now / 3)) < 1e-7:
        for i in range(ns):
            for c in range(nc):
                if visible[node_ids_t[i], c]:
                    observed[i, c] = True
                    known[i, c] = True
    for slot in range(slots):
        now = (global_slot + slot) * 0.005
        # packet_run의 'next_birth<duration' 조건은 구간 안에서는 항상 참이라 생략합니다.
        while next_birth <= now + 1e-10:
            pid = generated
            s = pid % ns
            d = (s + 1 + (pid // ns) % (ns - 1)) % ns
            source[pid] = s
            dest[pid] = d
            born[pid] = next_birth
            entry_time[pid] = next_birth
            first_tx[pid] = -1.
            if lengths[s] >= 64:
                status[pid] = 4
                end_time[pid] = next_birth
                overflow += 1
            else:
                row = pid % ring
                for col in range(64):
                    payload[row, col] = -1
                pos = cursor[s, d]
                col = 0
                checked = 0
                while checked < nc and col < 64:
                    if known[s, pos]:
                        payload[row, col] = pos
                        col += 1
                    pos = (pos + 1) % nc
                    checked += 1
                cursor[s, d] = pos
                queue[s, lengths[s]] = pid
                lengths[s] += 1
            generated += 1
            next_birth = generated / 30.0
        tx = (global_slot + slot) % n
        while lengths[tx] > 0 and now - born[queue[tx, 0]] >= 2.0 - 1e-10:
            pid = queue[tx, 0]
            status[pid] = 2
            end_time[pid] = now
            if first_tx[pid] >= 0:
                service_wait[pid] += now-first_tx[pid]
                service_before[pid] += max(0.,min(now,born[pid]+.25)-first_tx[pid])
            else:
                queue_wait[pid] += now-entry_time[pid]
                queue_before[pid] += max(0.,min(now,born[pid]+.25)-entry_time[pid])
            first_tx[pid] = -1.
            expire += 1
            pop_front(queue, lengths, tx)
        if (global_slot + slot) % 200 < reserved_slots:
            continue
        if lengths[tx] == 0:
            continue
        pid = queue[tx, 0]
        rx = routing[tx, dest[pid]]
        if rx < 0:
            if now <= born[pid]+.25+1e-10: evidence[pid,0] += 1
            frame_extra[tx,0] += 1
            continue
        if first_tx[pid] < 0:
            queue_wait[pid] += max(0.,now-entry_time[pid])
            queue_before[pid] += max(0.,min(now,born[pid]+.25)-entry_time[pid])
            first_tx[pid] = now
        d = dest[pid]
        if last_next[tx,d] >= 0 and last_next[tx,d] != rx: frame_extra[tx,1] += 1
        last_next[tx,d] = rx
        last_next_time[tx,d] = now
        if tries[pid] > 0: frame_extra[tx,2] += 1
        attempts += 1
        attempt_out[tx, rx] += 1
        if tries[pid] > 0:
            retries += 1
        tries[pid] += 1
        draw = np.random.random()
        if draw >= current[tx,rx] and now <= born[pid]+.25+1e-10:
            evidence[pid,1] += 1
            if draw < no_jam[tx,rx]: evidence[pid,2] += 1
            if blocked[tx,rx]: evidence[pid,3] += 1
            else: evidence[pid,4] += 1
        if draw < current[tx, rx]:
            service_wait[pid] += now+.005-first_tx[pid]
            service_before[pid] += max(0.,min(now+.005,born[pid]+.25)-first_tx[pid])
            first_tx[pid] = -1.
            frame_extra[tx,3] += 1
            successful += 1
            hops[pid] += 1
            success_out[tx, rx] += 1
            last_ack_out[tx, rx] = now + 0.005
            if tx >= ns:
                relay_hops[pid] += 1
                relay_successes += 1
            pop_front(queue, lengths, tx)
            tries[pid] = 0
            if rx == dest[pid]:
                status[pid] = 1
                end_time[pid] = now+.005
                delivered += 1
                delays[pid] = now + 0.005 - born[pid]
                if delays[pid] <= 0.25:
                    timely += 1
                last_received[source[pid], dest[pid]] = born[pid]
                for c in payload[pid % ring]:
                    if c >= 0:
                        known[rx, c] = True
            elif lengths[rx] >= 64:
                status[pid] = 4
                end_time[pid] = now+.005
                overflow += 1
            else:
                pos = lengths[rx]
                while pos > 0 and queue[rx, pos - 1] > pid:
                    queue[rx, pos] = queue[rx, pos - 1]
                    pos -= 1
                queue[rx, pos] = pid
                entry_time[pid] = now+.005
                lengths[rx] += 1
        elif tries[pid] >= 4:
            status[pid] = 3
            end_time[pid] = now+.005
            service_wait[pid] += now+.005-first_tx[pid]
            service_before[pid] += max(0.,min(now+.005,born[pid]+.25)-first_tx[pid])
            first_tx[pid] = -1.
            retrydrop += 1
            pop_front(queue, lengths, tx)
    global_slot += slots
    queue_out[:] = lengths
    union = 0
    intersection = 0
    for c in range(nc):
        u = False
        allk = True
        for i in range(ns):
            if observed[i, c]:
                u = True
            if not known[i, c]:
                allk = False
        if u:
            union += 1
        if allk:
            intersection += 1
    aoi = 0.0
    for i in range(ns):
        for j in range(ns):
            if i != j:
                aoi += (t + 1) * dt - last_received[i, j]
    aoi /= ns * (ns - 1)
    aoisum += aoi
    frame_out[:] = np.array(
        [
            generated,
            delivered,
            delivered / max(generated, 1),
            union / nc,
            intersection / nc,
            float(np.sum(lengths)),
            stat[0],
            stat[1],
            aoi,
            attempts,
            successful,
            expire,
            retrydrop,
            overflow,
            float(union),
            float(intersection),
            timely / max(generated, 1),
            relay_successes,
        ]
    )
    ic[0] = generated
    ic[1] = delivered
    ic[2] = attempts
    ic[3] = successful
    ic[4] = retries
    ic[5] = expire
    ic[6] = retrydrop
    ic[7] = overflow
    ic[8] = timely
    ic[9] = relay_successes
    ic[10] = global_slot
    ic[11] = outevents
    fc[0] = next_birth
    fc[1] = aoisum
    fc[2] = outage
    fc[3] = outagemax
    fc[4] = outagelen
    return current, stat
