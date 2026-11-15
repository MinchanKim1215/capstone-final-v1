"""지도·가중치 없이 실행하는 단위 검사.

  python -m unittest final_v1_relocation.tests.test_relocation -v

1. 모든 개선안을 끈 판단기가 final_v1/control.py의 choose와 같은 결정을 내는지
2. 개선안 1: 수색기 비콘이 끊겨 V18이 멈춘 상황에서 복구 명령을 내는지
3. 개선안 2: 대응 방식끼리 같은 수신 표본을 쓰고, 6초 뒤 기본 대형으로 돌아가는지
4. 개선안 3: 낮춘 문턱과 애매한 구간의 작은 이동 제한
5. 개선안 4: 차폐된 단일 경로에서 우회 경로를 만드는지
6. 실행기 연결(patch)과 평가 지표
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from adaptive_joint_expanded.features import OFFSETS, widest
from final_v1 import control as v21
from final_v1 import runner as native
from forecast_avoidance.controller import TIMES, samples
from forecast_prevention.controller import RadioState

from final_v1_relocation import bypass as bypass_module
from final_v1_relocation.control import Controller
from final_v1_relocation.evaluation import compare, outage_profile
from final_v1_relocation.graph import closure, first_sustained
from final_v1_relocation.options import ARMS, Options
from final_v1_relocation.runner import run as relocation_run
from final_v1_relocation.transport import FallbackChannel

ACTION = {tuple(o): i for i, o in enumerate(OFFSETS.astype(int).tolist())}


def radio_state(n, known=None, age=None, noise=None, local_rf=True):
    known = np.ones(n, bool) if known is None else np.asarray(known, bool)
    return RadioState(np.zeros((n, 9, 2)), known, np.zeros(n) if age is None else np.asarray(age, float),
                      np.zeros(n), np.full((n, n), np.nan), np.full(n, -107.) if noise is None else np.asarray(noise),
                      np.zeros(n), False, False, local_rf)


# ---------------------------------------------------------------- V21 대조용 가짜 입력
class Scenario:
    """후보마다 결정적인 무작위 여유·그래프·도착점을 돌려주는 extract 대역."""

    def __init__(self, seed, n=6, ns=3, base_actions=None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.n, self.ns = n, ns
        self.base = np.asarray(base_actions)
        self.base_margin = np.full((2, 18), 3.)
        self.base_margin[:, 6:15] = self.rng.uniform(-2., 1., size=(2, 9))
        self.base_margin[:, 0] = self.rng.choice([2., -1.], p=[.9, .1])

    def extract(self, planner, candidate, local, phase):
        candidate = np.asarray(candidate)
        rng = np.random.default_rng([self.seed, *candidate.tolist()])
        same = np.array_equal(candidate, self.base)
        high = 4. if self.seed % 3 else .5        # 세 번에 한 번은 개선 후보가 거의 없는 상황
        margins = self.base_margin if same else self.base_margin+rng.uniform(-1.2, high, size=(2, 18))
        graph = rng.normal(0., 6., size=(2, 18, self.n, self.n))
        graph = np.minimum(graph, graph.swapaxes(-1, -2))
        endpoint = np.zeros((self.n, 2)) if same or rng.random() < .1 else rng.normal(size=(self.n, 2))
        warning = SimpleNamespace(shield_score=float(rng.uniform(0, .5)), onset_in_s=4.5)
        return rng.normal(size=8).astype(np.float32), warning, margins, graph, endpoint


def fake_planner(n=6, ns=3, local_rf=True):
    return SimpleNamespace(state=radio_state(n, local_rf=local_rf), design=SimpleNamespace(n_search=ns))


class V21Equivalence(unittest.TestCase):
    def run_both(self, seed, score, arm="v21", phase=0, v18=None, local_rf=True, options=None):
        rng = np.random.default_rng(seed+1)
        base = rng.integers(0, 25, size=3)
        valid = rng.random((3, 25)) < .6
        valid[:, 0] = True
        valid[np.arange(3), base] = True
        logits = rng.normal(size=(3, 25))
        world = Scenario(seed, base_actions=base)
        planner = fake_planner(local_rf=local_rf)
        v18_result = v18 if v18 is not None else (base, dict(reason="no_predicted_cut"))
        with patch.object(v21, "extract", world.extract), \
                patch.object(v21, "v18_choose", lambda *a: (v18_result[0].copy(), dict(v18_result[1]))), \
                patch.object(v21, "load_profile", lambda: ("learned", dict(threshold=.1775), None, None)), \
                patch.object(v21, "make_score", lambda *a: np.array([score])), \
                patch.object(v21, "short_score", lambda model, x: float(abs(x[0]) % .3)):
            old = v21.choose(arm, planner, base.copy(), logits, valid, None, phase, 30., None, True)
            new = Controller(options).choose(arm, planner, base.copy(), logits, valid, None, phase, 30., None, True)
        return old, new

    def assert_same(self, old, new):
        np.testing.assert_array_equal(old[0], new[0])
        self.assertEqual(old[2], new[2])
        if old[1] is None:
            self.assertIsNone(new[1])
        else:
            self.assertEqual(old[1], {k: new[1][k] for k in old[1]})

    def test_default_controller_reproduces_v21(self):
        outcomes = set()
        for seed in range(300):
            score = (.05, .16, .2, .4)[seed % 4]
            arm = ("v21", "v21", "no_backup", "reactive", "physics_trigger")[seed % 5]
            old, new = self.run_both(seed, score, arm=arm, phase=2 if seed % 37 == 0 else 0,
                                     local_rf=seed % 29 != 0)
            self.assert_same(old, new)
            outcomes.add(old[2])
        # 무작위 상황이 V21의 주요 분기를 모두 지났는지 확인
        for reason in ("below_advance_threshold", "added_prevention", "no_robust_safe_gain",
                       "no_physical_prevention_evidence", "missing_rf_or_no_alternative", "no_recovery_action"):
            self.assertIn(reason, outcomes)

    def test_v18_branches_are_unchanged(self):
        base = np.array([0, 0, 0])
        moved = np.array([0, 3, 0])
        for reason in ("recovery", "prevention"):
            detail = dict(reason=reason, projected_fleet_endpoint=[[0., 0.]], changed_relay=4)
            for arm in ("v21", "reactive"):
                for allow in (True, False):
                    planner = fake_planner()
                    with patch.object(v21, "v18_choose", lambda *a: (moved.copy(), dict(detail))):
                        old = v21.choose(arm, planner, base.copy(), np.zeros((3, 25)), np.ones((3, 25), bool),
                                         None, 2, 30., None, allow)
                        new = Controller().choose(arm, planner, base.copy(), np.zeros((3, 25)), np.ones((3, 25), bool),
                                                  None, 2, 30., None, allow)
                    self.assert_same(old, new)

    # ------------------------------------------------------------ 개선안 3
    def test_lower_threshold_acts_between_thresholds(self):
        acted = 0
        for seed in range(200):
            old, new = self.run_both(seed, .15, options=Options(threshold_scale=.8))
            self.assertIn(old[2], ("below_advance_threshold", "missing_rf_or_no_alternative"))
            if new[2] == "added_prevention":
                acted += 1
                self.assertTrue(new[1]["ambiguous_band"])
                self.assertAlmostEqual(new[1]["trigger_threshold"], .1775*.8)
            # 점수가 낮춘 문턱(0.142)보다 낮으면 여전히 움직이지 않는다
            _, low = self.run_both(seed, .13, options=Options(threshold_scale=.8))
            self.assertIn(low[2], ("below_advance_threshold", "missing_rf_or_no_alternative"))
        self.assertGreater(acted, 20)

    def test_ambiguous_band_allows_only_small_moves(self):
        checked = 0
        for seed in range(200):
            _, new = self.run_both(seed, .15, options=Options(threshold_scale=.8, ambiguous_small_moves=True))
            if new[2] == "added_prevention":
                event = new[1]
                change = np.linalg.norm(OFFSETS[event["action"]]-OFFSETS[event["base_action"]])
                self.assertLessEqual(change, 28.3+1e-6)
                checked += 1
            # 원래 문턱 이상이면 큰 이동도 허용한다 (V21과 같다)
            old, high = self.run_both(seed, .3, options=Options(threshold_scale=.8, ambiguous_small_moves=True))
            np.testing.assert_array_equal(old[0], high[0])
        self.assertGreater(checked, 5)


# ---------------------------------------------------------------- 개선안 1
class GeoPlanner:
    """직선 거리로 링크 여유를 정하는 작은 기하 모델: 여유 = 15 - 0.15 × 거리(m) - 잡음 상승."""

    def __init__(self, start, state, ns):
        self.start = np.asarray(start, float)
        self.state = state
        self.design = SimpleNamespace(n_search=ns, n_drones=len(start))

    def path(self, actions):
        knots = np.repeat(self.start[None], 4, axis=0)
        move = np.zeros_like(self.start)
        move[self.design.n_search:] = OFFSETS[actions]
        knots[1:] += move
        return knots

    def radio(self, paths, trend=False):
        xy = np.stack([samples(p) for p in paths])
        distance = np.linalg.norm(xy[..., :, None, :]-xy[..., None, :, :], axis=-1)
        noise = np.maximum(self.state.noise[:, None], self.state.noise[None, :])+107.
        graph = 15.-.15*distance-noise
        idx = np.arange(graph.shape[-1])
        graph[..., idx, idx] = 30.
        return widest(graph, self.design.n_search).min(axis=-1), graph


START = [(0, 0), (60, 0), (300, 0), (30, 20), (150, 0), (180, 0)]   # 수색기 0~2, 중계기 3~5


def lost_case(age=6., relay_known=True):
    known = np.ones(6, bool)
    known[2] = False                      # 수색기 2의 비콘이 3초 넘게 끊김
    known[5] = relay_known
    state = radio_state(6, known=known, age=[0, 0, age, 0, 0, 0])
    return GeoPlanner(START, state, 3)


class CutRecovery(unittest.TestCase):
    def decide(self, planner, options):
        base = np.zeros(3, int)
        logits = np.zeros((3, 25))        # PPO 상위 후보에 끊긴 수색기 쪽 이동이 없도록 한다
        valid = np.ones((3, 25), bool)
        low = SimpleNamespace(shield_score=0., onset_in_s=np.nan)
        with patch.object(v21, "extract", lambda *a: (np.zeros(8), low, np.full((2, 18), 3.), None, np.zeros((6, 2)))), \
                patch.object(v21, "load_profile", lambda: ("learned", dict(threshold=.1775), None, None)), \
                patch.object(v21, "make_score", lambda *a: np.array([0.])):
            old = v21.choose("v21", planner, base.copy(), logits, valid, None, 0, 60., None, True)
            new = Controller(options).choose("v21", planner, base.copy(), logits, valid, None, 0, 60., None, True)
        return old, new

    def test_v21_skips_but_cut_recovery_reconnects(self):
        old, new = self.decide(lost_case(), Options(cut_recovery=True))
        self.assertEqual(old[2], "below_advance_threshold")          # 원래 V21: 복구 판단 없음
        self.assertEqual(new[2], "added_cut_recovery")
        event = new[1]
        self.assertEqual(event["reason"], "recovery")
        self.assertEqual(event["changed_relay"], 5)                 # 끊긴 수색기 쪽 중계기
        self.assertEqual(OFFSETS[event["action"]].tolist(), [40., 0.])
        self.assertIsNone(event["base_reconnect_in_s"])
        self.assertLess(event["projected_reconnect_in_s"], 4.)
        self.assertEqual(event["lost_search"], [2])

    def test_disabled_matches_v21(self):
        old, new = self.decide(lost_case(), Options())
        self.assertEqual(old[2], new[2])
        np.testing.assert_array_equal(old[0], new[0])

    def test_only_commandable_relays_move(self):
        _, new = self.decide(lost_case(relay_known=False), Options(cut_recovery=True))
        if new[1] is not None:
            self.assertNotEqual(new[1]["changed_relay"], 5)

    def test_long_cut_shrinks_move(self):
        _, new = self.decide(lost_case(age=20.), Options(cut_recovery=True))
        self.assertEqual(new[2], "added_cut_recovery")
        event = new[1]
        self.assertEqual(event["move_cap_m"], 40.)
        self.assertLessEqual(np.linalg.norm(OFFSETS[event["action"]]), 40.+1e-6)

    def test_model_evidence_skips_when_model_sees_connection(self):
        start = list(START)
        start[5] = (200, 0)                  # 보정 없는 모델 여유 = 15 - 0.15 x 100 = 0dB (끊김 아님)
        known = np.ones(6, bool)
        known[2] = False
        planner = GeoPlanner(start, radio_state(6, known=known, age=[0, 0, 6, 0, 0, 0]), 3)
        _, new = self.decide(planner, Options(cut_recovery=True, cut_evidence="model"))
        self.assertEqual(new[2], "below_advance_threshold")
        _, cut = self.decide(lost_case(), Options(cut_recovery=True, cut_evidence="model"))
        self.assertEqual(cut[2], "added_cut_recovery")
        self.assertLess(cut[1]["plain_reconnect_margin_db"], 0.)

    def test_persistent_evidence_needs_two_cycles(self):
        planner = lost_case()
        base = np.zeros(3, int)
        low = SimpleNamespace(shield_score=0., onset_in_s=np.nan)
        controller = Controller(Options(cut_recovery=True, cut_evidence="persistent"))
        with patch.object(v21, "extract", lambda *a: (np.zeros(8), low, np.full((2, 18), 3.), None, np.zeros((6, 2)))), \
                patch.object(v21, "load_profile", lambda: ("learned", dict(threshold=.1775), None, None)), \
                patch.object(v21, "make_score", lambda *a: np.array([0.])):
            first = controller.choose("v21", planner, base.copy(), np.zeros((3, 25)), np.ones((3, 25), bool), None, 0, 60., None, True)
            second = controller.choose("v21", planner, base.copy(), np.zeros((3, 25)), np.ones((3, 25), bool), None, 0, 63., None, True)
        self.assertEqual(first[2], "below_advance_threshold")
        self.assertEqual(controller.counts["cut_recovery:no_persistent_loss"], 1)
        self.assertEqual(second[2], "added_cut_recovery")
        self.assertTrue(second[1]["persistent_loss"])
        with self.assertRaises(ValueError):
            Options(cut_evidence="maybe")

    def test_projected_endpoint_comes_from_shared_plan(self):
        planner = lost_case()
        _, new = self.decide(planner, Options(cut_recovery=True))
        chosen = new[0]
        np.testing.assert_array_equal(new[1]["projected_fleet_endpoint"], planner.path(chosen)[1].tolist())


# ---------------------------------------------------------------- 개선안 2
class FakeWorld:
    def __init__(self):
        self.city = SimpleNamespace(points=np.array([[0., 0.], [50., 0.], [100., 0.], [0., 30.], [40., 30.],
                                                     [80., 30.], [120., 30.], [160., 30.]]))
        self.nodes = np.array([0, 1, 2, 3, 4, 5])
        self.proposal = dict(nominal=self.city.points[self.nodes].copy())
        self.scenario = SimpleNamespace(grid_m=10.)


def channel_run(mode, seed=11, ticks=6):
    env = FakeWorld()
    channel = FallbackChannel(seed, 6, 3, 10_000, mode, lambda: env)
    observation = dict(action_mask=np.ones((3, 25), bool), teacher=np.zeros((3, 25)))
    requested = np.array([ACTION[(20, 0)], ACTION[(0, 20)], ACTION[(-40, 0)]])
    out = []
    for tick in range(ticks):
        p = np.ones((6, 6))
        if tick >= 1:
            p[5, :] = p[:, 5] = 0.                    # 1틱부터 중계기 2(드론 5)가 명령을 못 받는다
            env.nodes = np.array([0, 1, 2, 3, 4, 7])   # 그 사이 중계기 2는 오른쪽으로 밀려났다
            env.proposal = dict(nominal=env.city.points[[0, 1, 2, 3, 4, 7]].copy())
        out.append(channel.apply(tick, p, requested, observation, np.ones(3, bool), "local"))
    return out, requested


class CommandFallback(unittest.TestCase):
    def test_common_random_numbers(self):
        runs = {mode: channel_run(mode)[0] for mode in ("neutral", "hold", "reconnect", "local")}
        for tick in range(6):
            for mode in ("hold", "reconnect", "local"):
                np.testing.assert_array_equal(runs["neutral"][tick][1], runs[mode][tick][1])
                np.testing.assert_array_equal(runs["neutral"][tick][3], runs[mode][tick][3])

    def test_reconnect_returns_then_times_out(self):
        out, requested = channel_run("reconnect")
        self.assertTrue(out[0][1][2])                    # 0틱: 받음 → 위치 (80, 30)을 기억
        anchor = np.array([80., 30.])
        nominal = np.array([160., 30.])
        for tick in (1, 2):                              # 6초 동안은 마지막 수신 위치 쪽
            self.assertFalse(out[tick][1][2])
            self.assertEqual(out[tick][4][2], "return_anchor")
            target = nominal+OFFSETS[out[tick][0][2]]
            self.assertLess(np.linalg.norm(target-anchor), np.linalg.norm(nominal-anchor))
            self.assertEqual(OFFSETS[out[tick][0][2]].tolist(), [-40., 0.])
        for tick in (3, 4, 5):                           # 그 뒤에는 기본 대형
            self.assertEqual(out[tick][4][2], "nominal_timeout")
            self.assertEqual(out[tick][0][2], 0)

    def test_hold_keeps_last_command_for_window(self):
        out, requested = channel_run("hold")
        for tick in (1, 2):
            self.assertEqual(out[tick][4][2], "hold_last")
            self.assertEqual(out[tick][0][2], requested[2])
        for tick in (3, 4):
            self.assertEqual(out[tick][0][2], 0)

    def test_neutral_matches_final_v1_channel(self):
        from final_v1.transport import CommandChannel

        env = FakeWorld()
        ours = FallbackChannel(5, 6, 3, 40, "neutral", lambda: env)
        theirs = CommandChannel(5, 6, 3, 40)
        observation = dict(action_mask=np.ones((3, 25), bool), teacher=np.zeros((3, 25)))
        rng = np.random.default_rng(3)
        for tick in range(30):
            p = rng.uniform(.0, 1., size=(6, 6))
            requested = rng.integers(0, 25, size=3)
            a = ours.apply(tick, p, requested, observation, np.ones(3, bool), "local")
            b = theirs.apply(tick, p, requested, observation, np.ones(3, bool), "neutral")
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)


# ---------------------------------------------------------------- 개선안 4
def chain_graph(redundant):
    g = np.full((2, 18, 4, 4), -10.)
    for i, j in ((0, 2), (2, 1)) + (((0, 3), (3, 1)) if redundant else ()):
        g[..., i, j] = g[..., j, i] = 5.
    idx = np.arange(4)
    g[..., idx, idx] = 30.
    return g


class Bypass(unittest.TestCase):
    def decide(self, options, shield=12.):
        planner = SimpleNamespace(state=radio_state(4), design=SimpleNamespace(n_search=2))
        base = np.zeros(2, int)
        margins = np.full((2, 18), 5.)

        def extract(planner, candidate, local, phase):
            redundant = candidate[1] != 0                   # 중계기 1(드론 3)을 움직이면 두 번째 경로가 생긴다
            endpoint = np.zeros((4, 2)) if not candidate.any() else np.full((4, 2), float(candidate.sum()))
            return np.zeros(8), SimpleNamespace(shield_score=0., onset_in_s=np.nan), margins, \
                chain_graph(redundant), endpoint

        extra = np.zeros((18, 4, 4))
        extra[:, 2, 1] = extra[:, 1, 2] = shield            # 중계기 0 ↔ 수색기 1 링크가 건물에 가림
        with patch.object(v21, "extract", extract), \
                patch.object(v21, "load_profile", lambda: ("learned", dict(threshold=.1775), None, None)), \
                patch.object(v21, "make_score", lambda *a: np.array([0.])), \
                patch.object(v21, "v18_choose", lambda *a: (base.copy(), dict(reason="no_predicted_cut"))), \
                patch.object(bypass_module, "building_extra", lambda *a: extra):
            controller = Controller(options)
            first = controller.choose("v21", planner, base.copy(), np.zeros((2, 25)), np.ones((2, 25), bool),
                                      None, 0, 30., None, True)
            again = controller.choose("v21", planner, base.copy(), np.zeros((2, 25)), np.ones((2, 25), bool),
                                      None, 0, 33., None, True)
        return first, again

    def test_shielded_single_path_gets_backup(self):
        (actions, event, reason), again = self.decide(Options(bypass=True))
        self.assertEqual(reason, "added_bypass")
        self.assertEqual(event["changed_relay"], 3)
        self.assertGreater(event["single_link_margin_db"], event["base_single_link_margin_db"])
        self.assertEqual(event["shield_exposure_db"], 12.)
        self.assertEqual(again[2], "below_advance_threshold")   # 6초 대기 시간

    def test_no_shielding_no_move(self):
        (_, event, reason), _ = self.decide(Options(bypass=True), shield=0.)
        self.assertIsNone(event)
        self.assertEqual(reason, "below_advance_threshold")

    def test_disabled(self):
        (_, event, reason), _ = self.decide(Options())
        self.assertIsNone(event)


# ---------------------------------------------------------------- 실행기와 지표
class Wiring(unittest.TestCase):
    def test_runner_patches_and_restores(self):
        original_choose, original_channel, original_world = native.choose, native.CommandChannel, native.World
        seen = {}

        def fake_run(spec, arm, transport=None, max_steps=None):
            seen["choose"] = native.choose
            seen["channel"] = native.CommandChannel(1, 6, 3, 40) if transport else None
            return dict(spec=spec), {}

        with patch.object(native, "run", fake_run):
            row, _ = relocation_run(dict(region="x"), "r1_cut_recovery", transport="reconnect")
        self.assertEqual(seen["choose"].__self__.options, ARMS["r1_cut_recovery"])
        self.assertIsInstance(seen["channel"], FallbackChannel)
        self.assertEqual(seen["channel"].mode, "reconnect")
        self.assertIs(native.choose, original_choose)
        self.assertIs(native.CommandChannel, original_channel)
        self.assertIs(native.World, original_world)
        self.assertEqual(row["relocation_arm"], "r1_cut_recovery")
        with self.assertRaises(ValueError):
            relocation_run(dict(region="x"), "unknown")

    def test_graph_helpers(self):
        rng = np.random.default_rng(0)
        g = rng.normal(size=(5, 7, 7))
        g = np.minimum(g, g.swapaxes(-1, -2))
        i, j = np.triu_indices(3, 1)
        np.testing.assert_allclose(closure(g)[..., i, j], widest(g, 3))
        margin = np.full((2, 18), -1.)
        margin[1, 5:] = 1.
        self.assertEqual(first_sustained(margin).tolist(), [np.inf, TIMES[5]])

    def test_outage_metrics(self):
        def row(region, cuts, onsets):
            return dict(spec=dict(region=region, kind="clear", seed=1, direction="east_west"), duration_s=3600.,
                        persistent_cut_intervals=cuts, persistent_cut_seconds=sum(b-a for a, b in cuts),
                        persistent_onsets=onsets, generated_packets=100, timely_pdr=.9, mission_completed=True,
                        actual_safety_violations=0, relay_travel_m=1000.)
        ref = [row("a", [[0., 4.], [10., 30.], [50., 52.]], 2), row("b", [[5., 7.]], 1)]
        alt = [row("a", [[0., 4.], [10., 14.]], 1), row("b", [], 0)]
        profile = outage_profile(ref)
        self.assertEqual(profile["new_outages"], 3)
        self.assertEqual(profile["long_outages"], 1)
        self.assertAlmostEqual(profile["long_share_of_cut_time"], 20./28.)
        change = compare(ref, alt, lambda r: r["spec"]["region"], draws=200)
        self.assertAlmostEqual(change["starts_per_h_delta"], -1.)
        self.assertAlmostEqual(change["long_per_h_delta"], -.5)


    def test_campaign_summary_and_report(self):
        import json
        import tempfile
        from pathlib import Path

        from final_v1_relocation import campaign
        from final_v1_relocation.report import markdown

        rng = np.random.default_rng(0)
        folder = Path(tempfile.mkdtemp())

        def row(arm, region, transport=None):
            cuts = [[0., 2.]]+[[float(a), float(a+rng.choice([1.5, 4., 12.]))]
                               for a in sorted(rng.uniform(30, 1700, rng.integers(1, 6)))]
            network = np.ones(3600, bool)
            for a, b in cuts:
                network[int(a*2):int(b*2)] = False
            path = folder/f"{arm}_{region}_{transport}.npz"
            np.savez(path, network=network, received=rng.random((600, 6)) < .95,
                     fallback=np.full((600, 6), "received"))
            return dict(spec=dict(region=region, kind="clear", seed=1, direction="east_west"), relocation_arm=arm,
                        transport=transport, duration_s=1800., persistent_cut_intervals=cuts,
                        persistent_cut_seconds=sum(b-a for a, b in cuts), persistent_onsets=len(cuts)-1,
                        generated_packets=1000, timely_pdr=.97, mission_completed=True, actual_safety_violations=0,
                        relay_travel_m=32000., delay_mean_s=.2, delay_p95_s=.6,
                        intervention=[dict(reason="recovery", expert="cut_recovery", actual_pre_action_connected=False,
                                           executed_as_requested=True)],
                        v18_reasons={"insufficient_observation": 10, "no_predicted_cut": 500},
                        relocation_counts={}, decision_counts={}, archive_path=str(path))

        regions = ["a", "b", "c"]
        confirm_rows = [row(arm, r) for arm in ("v21", "r1_cut_recovery") for r in regions]
        stress_rows = [row(a, r, f) for a, f in (("v21", "neutral"), ("v21", "reconnect")) for r in regions]
        with patch.object(campaign, "group_lookup", lambda: (lambda r: r["spec"]["region"])):
            _, confirm = campaign.evaluate_confirm(confirm_rows, 3)
            stress = campaign.evaluate_stress(stress_rows, 3)
        result = dict(controller="v21", started_at="s", finished_at="f", steps=None, limit=None,
                      confirmation=confirm, command_loss=stress)
        json.dumps(result, allow_nan=False)
        text = markdown(result)
        self.assertIn("개선안 1 · 단절 중 복구", text)
        self.assertIn("마지막 수신 위치 복귀", text)
        self.assertEqual(confirm["stats"]["r1_cut_recovery"]["cut_recovery_commands"], 3)


    def test_regenerated_specs_match_final_v1_replay(self):
        from final_v1_relocation import campaign
        from final_v1_relocation.replay_audit import load

        regions = sorted({c["region"] for c in load()["cases"] if c["phase"] == "비교"})
        with patch("forecast_rebuild.contracts.folds", lambda: dict(confirmation=regions)):
            confirm, stress = campaign.regenerate()
        self.assertEqual((len(confirm), len(stress)), (42, 14))
        with self.assertRaises(ValueError):
            campaign.check_against_replay(confirm[1:], stress)

    def test_freeze_paths_are_localized(self):
        import forecast_learning.control as learned
        from forecast_learning import ARTIFACTS as LEARNED
        from final_v1_relocation.runner import localize_freeze

        fake = dict(models=dict(early=dict(path="C:\\Users\\team\\forecast_learning\\artifacts\\models\\dual_early.txt")))
        with patch.object(learned, "read", lambda path: json_copy(fake)), \
                patch("pathlib.Path.is_file", lambda self: str(self).startswith(str(LEARNED))):
            localize_freeze()
            value = learned.read(LEARNED/"freeze.json")
        self.assertEqual(value["models"]["early"]["path"], str(LEARNED/"models"/"dual_early.txt"))

    def test_absolute_map_paths_are_found_locally(self):
        import tempfile
        from pathlib import Path

        from adaptive_joint_expanded import contracts
        from final_v1_relocation import paths

        root = Path(tempfile.mkdtemp())
        raw = root/"additional_maps_20261030"/"raw"/"batch2"/"maps"/"incheon_x"/"scene.local.json"
        raw.parent.mkdir(parents=True)
        raw.write_text("{}")
        moved = root/"drone_forecast_joint_training"/"maps"/"elsewhere"/"gyeonggi_y"/"scene.local.json"
        moved.parent.mkdir(parents=True)
        moved.write_text("{}")
        kept = root/"drone_forecast_joint_training"/"maps"/"additional"/"seoul_z"/"scene.local.json"
        kept.parent.mkdir(parents=True)
        kept.write_text("{}")
        author = "C:\\Users\\author\\OneDrive\\캡스톤\\additional_maps_20261030\\raw\\batch2\\maps\\incheon_x\\scene.local.json"
        value = dict(maps=[dict(key="incheon_x", path=author),
                           dict(key="gyeonggi_y", path="C:\\Users\\author\\maps\\gyeonggi_y\\scene.local.json"),
                           dict(key="seoul_z", path="maps/additional/seoul_z/scene.local.json"),
                           dict(key="missing", path="C:\\nowhere\\missing\\scene.local.json")])
        with patch.object(paths, "WORKSPACE", root), patch.object(contracts, "plan", lambda: json_copy(value)):
            paths.localize_maps()
            result = {m["key"]: m["path"] for m in contracts.plan()["maps"]}
        self.assertEqual(result["incheon_x"], str(raw))
        self.assertEqual(result["gyeonggi_y"], str(moved))
        self.assertEqual(result["seoul_z"], "maps/additional/seoul_z/scene.local.json")
        self.assertTrue(result["missing"].startswith("C:"))


def json_copy(value):
    import json

    return json.loads(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
