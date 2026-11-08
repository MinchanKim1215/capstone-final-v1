"""기존 전체 엔진(fast_network)의 채널·패킷 계산을 한 구간씩 나눠 실행하는 부품.

기존 usable.collect는 임무 전체 궤적을 먼저 만든 뒤 통신을 일괄 계산합니다.
재배치 학습에는 '3초 이동 → 그 구간 통신 → 다음 행동'의 순서가 필요하므로,
같은 수식과 같은 난수 순서를 유지한 채 구간 단위로 쪼갰습니다.

- shadow_sequence: probabilities()와 같은 순서로 섀도잉 난수를 미리 뽑습니다.
  섀도잉 난수는 드론 위치와 무관하므로 미리 뽑아도 결과가 같습니다.
- hop_probabilities: probabilities()의 한 시각 계산과 같은 식입니다.
- packet_interval / packet_drain: packet_run()의 반복문 한 번(한 구간)과 종료 정산입니다.
- NumbaRNG: Numba 전역 난수 상태를 환경마다 따로 저장·복원합니다.
  여러 환경을 한 프로세스에서 번갈아 돌려도 서로의 패킷 난수가 섞이지 않습니다.

보정 0의 전체 임무 일치: cooperative.verify. 스트리밍 패킷 일치: tests/test_cooperative.py.
"""

import math
import numpy as np
from numba import njit, _helperlib
from swarm_sim.fast_network import bpsk_probability, graph_statistics, pop_front

# 패킷 페이로드 원형 버퍼 크기입니다. 패킷은 최대 약 2.04초만 살아 있으므로(30개/초)
# 동시에 살아 있는 패킷은 100개 미만입니다. 4096이면 겹쳐 쓰는 일이 없습니다.
PAYLOAD_RING = 4096


@njit(cache=True)
def _numba_seed(seed):
    np.random.seed(seed)


class NumbaRNG:
    """Numba 전역 np.random 상태를 이 객체 전용으로 보관합니다."""

    def __init__(self, seed):
        _numba_seed(seed)
        self.state = _helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())

    def __enter__(self):
        _helperlib.rnd_set_state(_helperlib.rnd_get_np_state_ptr(), self.state)
        return self

    def __exit__(self, *exc):
        self.state = _helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())
        return False


# probabilities()의 섀도잉 갱신만 떼어 냈습니다. 난수 추출 순서(t → i → j<i)가 같습니다.
@njit(cache=True)
def shadow_sequence(seed, nt, n, dt, sigma, tau):
    np.random.seed(seed)
    out = np.zeros((nt, n, n))
    shadow = np.zeros((n, n))
    rho = math.exp(-dt / tau)
    sd = sigma * math.sqrt(1 - rho * rho)
    for t in range(nt):
        for i in range(n):
            for j in range(i):
                shadow[i, j] = rho * shadow[i, j] + sd * np.random.normal()
                shadow[j, i] = shadow[i, j]
        out[t] = shadow
    return out


# probabilities()의 한 시각 계산과 같은 식입니다. received/sinr는 예측 입력용 기록입니다.
@njit(cache=True)
def hop_probabilities(loss, interference, shadow, received_out, sinr_out):
    n = loss.shape[0]
    p = np.zeros((n, n))
    rx = np.zeros((n, n))
    den = np.zeros(n)
    for j in range(n):
        den[j] = 10.0 * math.log10(10 ** (-107 / 10) + interference[j])
    for i in range(n):
        for j in range(n):
            rx[i, j] = 10.0 - loss[i, j] - shadow[i, j] - den[j]
            received_out[i, j] = 10.0 - loss[i, j] - shadow[i, j]
            sinr_out[i, j] = rx[i, j]
    for i in range(n):
        for j in range(n):
            if i != j:
                p[i, j] = bpsk_probability(rx[i, j], 4608) * bpsk_probability(
                    rx[j, i], 112
                )
    return p


# packet_run()의 'for t in range(nt)' 반복 한 번입니다.
# ic: 0생성 1전달 2시도 3성공홉 4재시도 5수명초과 6재시도한도 7큐초과 8적시 9중계성공 10전역슬롯 11단절사건
# fc: 0다음생성시각 1AoI합 2누적단절 3최장단절 4현재단절길이
@njit(cache=True)
def packet_interval(
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
):
    n = p_t.shape[0]
    nc = visible.shape[1]
    slots = int(round(dt / 0.005))
    ring = payload.shape[0]
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
            if lengths[s] >= 64:
                status[pid] = 4
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
            expire += 1
            pop_front(queue, lengths, tx)
        if (global_slot + slot) % 200 < reserved_slots:
            continue
        if lengths[tx] == 0:
            continue
        pid = queue[tx, 0]
        rx = routing[tx, dest[pid]]
        if rx < 0:
            continue
        attempts += 1
        attempt_out[tx, rx] += 1
        if tries[pid] > 0:
            retries += 1
        tries[pid] += 1
        if np.random.random() < current[tx, rx]:
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
                overflow += 1
            else:
                pos = lengths[rx]
                while pos > 0 and queue[rx, pos - 1] > pid:
                    queue[rx, pos] = queue[rx, pos - 1]
                    pos -= 1
                queue[rx, pos] = pid
                lengths[rx] += 1
        elif tries[pid] >= 4:
            status[pid] = 3
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


# packet_run()의 임무 종료 후 2초 정산과 남은 큐 처리입니다. current는 마지막 구간의 홉 확률입니다.
@njit(cache=True)
def packet_drain(
    current,
    ns,
    duration,
    known,
    payload,
    source,
    dest,
    born,
    status,
    tries,
    hops,
    relay_hops,
    delays,
    queue,
    lengths,
    ic,
):
    n = current.shape[0]
    ring = payload.shape[0]
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
    routing, _, _ = graph_statistics(current, ns)
    for slot in range(400):
        now = duration + slot * 0.005
        tx = (global_slot + slot) % n
        while lengths[tx] > 0 and now - born[queue[tx, 0]] >= 2.0 - 1e-10:
            pid = queue[tx, 0]
            status[pid] = 2
            expire += 1
            pop_front(queue, lengths, tx)
        if lengths[tx] == 0:
            continue
        pid = queue[tx, 0]
        rx = routing[tx, dest[pid]]
        if rx < 0:
            continue
        attempts += 1
        if tries[pid] > 0:
            retries += 1
        tries[pid] += 1
        if np.random.random() < current[tx, rx]:
            successful += 1
            hops[pid] += 1
            if tx >= ns:
                relay_hops[pid] += 1
                relay_successes += 1
            pop_front(queue, lengths, tx)
            tries[pid] = 0
            if rx == dest[pid]:
                status[pid] = 1
                delivered += 1
                delays[pid] = now + 0.005 - born[pid]
                if delays[pid] <= 0.25:
                    timely += 1
                for c in payload[pid % ring]:
                    if c >= 0:
                        known[rx, c] = True
            elif lengths[rx] >= 64:
                status[pid] = 4
                overflow += 1
            else:
                pos = lengths[rx]
                while pos > 0 and queue[rx, pos - 1] > pid:
                    queue[rx, pos] = queue[rx, pos - 1]
                    pos -= 1
                queue[rx, pos] = pid
                lengths[rx] += 1
        elif tries[pid] >= 4:
            status[pid] = 3
            retrydrop += 1
            pop_front(queue, lengths, tx)
    for node in range(n):
        for q in range(lengths[node]):
            status[queue[node, q]] = 2
            expire += 1
        lengths[node] = 0
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


class PacketSim:
    """packet_run()의 상태를 객체에 보관하고 구간 단위로 진행합니다."""

    def __init__(
        self,
        n,
        ns,
        visible,
        dt,
        seed,
        disable_relays=False,
        max_duration_s=7200.0,
        reserved_slots=8,
    ):
        self.n, self.ns, self.dt = n, ns, dt
        self.reserved_slots = int(reserved_slots)
        self.visible = visible
        self.disable_relays = bool(disable_relays)
        nc = visible.shape[1]
        capacity = int(math.ceil(max_duration_s * 30)) + 128
        self.known = np.zeros((ns, nc), np.bool_)
        self.observed = np.zeros((ns, nc), np.bool_)
        self.payload = np.full((PAYLOAD_RING, 64), -1, np.int32)
        self.source = np.zeros(capacity, np.int32)
        self.dest = np.zeros(capacity, np.int32)
        self.born = np.zeros(capacity)
        self.status = np.zeros(capacity, np.int8)
        self.tries = np.zeros(capacity, np.int32)
        self.hops = np.zeros(capacity, np.int32)
        self.relay_hops = np.zeros(capacity, np.int32)
        self.delays = np.zeros(capacity)
        self.last_received = np.zeros((ns, ns))
        self.cursor = np.zeros((ns, ns), np.int32)
        self.queue = np.full((n, 64), -1, np.int32)
        self.lengths = np.zeros(n, np.int32)
        self.ic = np.zeros(12, np.int64)
        self.fc = np.zeros(5)
        self.attempts = np.zeros((n, n), np.int32)
        self.successes = np.zeros((n, n), np.int32)
        self.last_ack = np.full((n, n), -1.0)
        self.queue_now = np.zeros(n, np.int32)
        self.frame = np.zeros(18)
        self.t_index = 0
        self.current = None
        self.finished = False
        self.rng = NumbaRNG(seed)

    # 한 구간(dt초) 진행. node_ids는 구간 시작의 격자 노드(센서 관측용)입니다.
    def interval(self, p_t, node_ids):
        if self.finished:
            raise RuntimeError("이미 종료 정산을 마친 패킷 시뮬레이터입니다.")
        if self.ic[0] >= len(self.source) - 64:
            raise RuntimeError(
                "패킷 배열 용량을 넘었습니다. max_duration_s를 늘리세요."
            )
        with self.rng:
            current, stat = packet_interval(
                np.ascontiguousarray(p_t, np.float64),
                np.ascontiguousarray(node_ids, np.int32),
                self.visible,
                self.ns,
                self.dt,
                self.t_index,
                self.disable_relays,
                self.known,
                self.observed,
                self.payload,
                self.source,
                self.dest,
                self.born,
                self.status,
                self.tries,
                self.hops,
                self.relay_hops,
                self.delays,
                self.last_received,
                self.cursor,
                self.queue,
                self.lengths,
                self.ic,
                self.fc,
                self.attempts,
                self.successes,
                self.last_ack,
                self.queue_now,
                self.frame,
                self.reserved_slots,
            )
        self.current = current
        self.t_index += 1
        return stat

    # 생성 중지 후 2초 정산. 이후 values()가 run_full의 packet_run 반환값과 같은 의미입니다.
    def drain(self):
        if self.finished:
            return
        if self.current is None:
            raise RuntimeError("진행한 구간이 없습니다.")
        with self.rng:
            packet_drain(
                self.current,
                self.ns,
                self.t_index * self.dt,
                self.known,
                self.payload,
                self.source,
                self.dest,
                self.born,
                self.status,
                self.tries,
                self.hops,
                self.relay_hops,
                self.delays,
                self.queue,
                self.lengths,
                self.ic,
            )
        self.finished = True

    # packet_run의 values 배열과 같은 순서입니다.
    def values(self):
        ic, fc = self.ic, self.fc
        return np.array(
            [
                ic[0],
                ic[1],
                ic[2],
                ic[3],
                ic[4],
                ic[5],
                ic[6],
                ic[7],
                ic[8],
                ic[9],
                fc[1] / max(self.t_index, 1),
                fc[2],
                fc[3],
                ic[11],
            ],
            dtype=float,
        )

    def generated_status(self):
        g = int(self.ic[0])
        return self.status[:g], self.delays[:g], self.hops[:g]
