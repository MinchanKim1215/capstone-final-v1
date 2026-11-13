"""Use the verified continuous-motion simulator with a joint forecast and actor."""

from unittest.mock import patch

import numpy as np
import torch

from fast_halfsec import actor, runner
from fast_halfsec.controller import PLANNER_DELAY_TICKS
from final_v1.transformer_upgrade.capture import N_GEOMETRY
from forecast_learning.control import GUARDS, load_profile, short_score
from forecast_learning.features import HELD_COLUMNS, PREFIX
from forecast_learning.training import make_score
from halfsec_choice.choice import branch

from .model import OBSERVATIONS
from .observations import History, ROUTE_TIMES, path_block, path_inputs


class JointFlight:
    def __init__(self, model, threshold, explore, comparison):
        self.model, self.threshold, self.explore, self.comparison = model, threshold, explore, comparison
        mode, self.profile, self.early, self.short = load_profile()
        if mode != 'learned':
            raise ValueError('The deployed V21 LightGBM profile is required')
        self.history = History()
        self.frames, self.scores, self.other_scores, self.support_counts = [], [], [], []
        self.geometry, self.support, self.route, self.native = [], [], [], []
        self.action_samples = []
        self.guard_rejections = 0
        self.survey_factory = runner.survey_for
        self.navigator_factory = runner.Navigator
        self.sensor_factory = runner.Sensors
        self.packet_factory = runner.PacketSim
        self.physical_state = runner.state_at
        self.physical_options = runner.options_for

    def survey(self, *args):
        result = self.survey_factory(*args)
        city, scenario, fleet, _, _ = result
        city._cooperative_scenario = scenario
        self.history.bind(city, fleet)
        return result

    def navigator(self, *args):
        self.nav = self.navigator_factory(*args)
        return self.nav

    def packet(self, *args, **kwargs):
        self.packets = self.packet_factory(*args, **kwargs)
        return self.packets

    def sensors(self, *args):
        sensor = self.sensor_factory(*args)
        deliver = sensor.deliver

        def observe(now):
            deliver(now)
            if now > 0. and abs(2. * now - round(2. * now)) < 1e-8:
                self.history.sample(now, self.nav, sensor, int(self.packets.lengths[self.nav.ns]))

        sensor.deliver = observe
        return sensor

    @torch.no_grad()
    def state_at(self, nav, scenario, now, applied, estimate, history):
        state = self.physical_state(nav, scenario, now, applied, estimate, history)
        queue = int(self.packets.lengths[nav.ns])
        x, route, shield, radio, release = path_inputs(nav, scenario, applied, estimate, self.history, queue)
        self.v21_x, self.radio, self.release = x, radio, release
        self.short_risk = short_score(self.short, x)
        native = np.asarray([make_score(self.early, self.profile, x[None], shield)[0],
                             self.short_risk], np.float32)
        observation = self.history.inputs()
        observation.update(geometry=observation['geometry'].astype(np.float16), route=route, native=native)
        args = [torch.from_numpy(observation[key][None]) for key in OBSERVATIONS]
        _, _, self.context = self.model.forecast(*args)
        self.frames.append(round(now * 2.))
        self.scores.append(float(self.context[0, 0]))
        self.support_counts.append(int(observation['support'].sum()))
        if self.comparison is not None:
            self.other_scores.append(float(self.comparison.forecast(*args)[2][0, 0]))
        if self.explore:
            for key in ('geometry', 'support', 'route', 'native'):
                getattr(self, key).append(observation[key])
        return state

    def predict(self, raw):
        return self.scores[-1], self.short_risk

    def options_for(self, nav, scenario, now, requested, applied, estimate, state, learned_risk, delivery):
        options, tested = self.physical_options(nav, scenario, now, requested, applied, estimate,
                                                state, learned_risk, delivery)
        if len(options) < 2:
            return options, tested
        baseline = branch(nav, scenario, estimate, applied, options[0], delivery)
        safe = [options[0]]
        for option in options[1:]:
            alternative = branch(nav, scenario, estimate, applied, option, PLANNER_DELAY_TICKS + delivery)
            if ((alternative.stages < baseline.stages).any()
                    or (alternative.blockers & ~baseline.blockers).any()):
                continue
            held = alternative.radio_xy[:len(ROUTE_TIMES)]
            features = self.v21_x.copy()
            features[len(PREFIX):len(HELD_COLUMNS)] = path_block(nav, scenario, held[:-1], self.radio)[0]
            moving = (held[6] != held[0]).any(-1)
            features[7:9] = moving[nav.ns:].mean(), moving[:nav.ns].mean()
            features[-2:] = np.linalg.norm(held[[12, 18]] - self.release[[12, 18]], axis=-1).mean(axis=1)
            short = short_score(self.short, features)
            if short > max(GUARDS['short_score_floor'], self.short_risk + GUARDS['short_score_tolerance']):
                continue
            option.event.update(v21_short_before=self.short_risk, v21_short_after=short)
            safe.append(option)
        self.guard_rejections += len(options) - len(safe)
        return safe, tested

    @torch.no_grad()
    def select(self, policy, inputs, explore, rng):
        args = [torch.from_numpy(part[None]) for part in inputs]
        logits, value = self.model.policy(*args, self.context)
        prob = logits[0].softmax(dim=-1).numpy().astype(np.float64)
        action = int(rng.choice(len(prob), p=prob/prob.sum())) if explore else int(logits[0].argmax())
        if explore:
            self.action_samples.append(len(self.frames) - 1)
        return action, float(value[0]), float(logits[0].log_softmax(dim=-1)[action])

    def run(self, spec, cadence, max_seconds):
        if cadence not in (.5, 3.) or (self.explore and cadence != .5):
            raise ValueError('Joint exploration requires the half-second clock')
        mode = 'physics_3s' if cadence == 3. else 'ppo_05s'
        with (patch.multiple(runner, survey_for=self.survey, Navigator=self.navigator,
                             Sensors=self.sensors, PacketSim=self.packet, state_at=self.state_at,
                             options_for=self.options_for),
              patch.object(actor, 'select', self.select)):
            row, arrays = runner.full_mission(spec, mode, models=self, policy=self.model,
                                              explore=self.explore, max_seconds=max_seconds)
        row.update(decision_interval_s=cadence, transformer_evaluations=sum(n > 0 for n in self.support_counts),
                   guard_rejections=self.guard_rejections, joint_actor_decisions=len(self.action_samples))
        arrays.update(plan_frames=np.asarray(self.frames, np.int32), scores=np.asarray(self.scores, np.float32),
                      comparison_scores=np.asarray(self.other_scores, np.float32),
                      support_counts=np.asarray(self.support_counts, np.int16))
        if self.explore:
            np.testing.assert_array_equal(arrays['action_frames'], arrays['plan_frames'][self.action_samples] * 5)
            arrays.update(action_samples=np.asarray(self.action_samples, np.int32),
                          history_values=np.asarray(self.history.values, np.float16),
                          history_masks=np.asarray(self.history.masks, bool),
                          history_intents=np.asarray(self.history.intents, np.float16))
            nr, n = self.history.deployment.nr, self.history.deployment.n
            arrays.update(geometry=np.asarray(self.geometry, np.float16).reshape(-1, nr, n, 46),
                          support=np.asarray(self.support, bool).reshape(-1, nr, n),
                          route=np.asarray(self.route, np.float32).reshape(-1, N_GEOMETRY),
                          native=np.asarray(self.native, np.float32).reshape(-1, 2))
        return row, arrays
