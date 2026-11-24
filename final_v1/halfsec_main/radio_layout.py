"""Evaluate constant airborne elevations and an extra fixed relay in full missions."""

from copy import copy
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from fast_halfsec.navigation import GUARD
from swarm_sim.mission import MAX_MISSION

from .mission import MainFlight
from .release import load_model, map_scope


@dataclass(frozen=True)
class RadioLayout:
    elevated_relay: int | None = None
    height_m: float | None = None
    fixed_node: int | None = None


def station_graph(city, node):
    """Reserve the stationary relay's separation envelope on the flight graph."""
    edges = city.graph.tocoo()
    source = city.points[edges.row]
    step = city.points[edges.col] - source
    offset = city.points[node] - source
    fraction = np.clip(np.sum(offset * step, axis=1) / (edges.data ** 2), 0., 1.)
    close = source + fraction[:, None] * step
    keep = np.linalg.norm(close - city.points[node], axis=1) >= GUARD
    return csr_matrix((edges.data[keep], (edges.row[keep], edges.col[keep])), shape=edges.shape)


def mobile_component(city, survey, ns, node):
    if not 0 <= node < len(city.points):
        raise ValueError('Station is outside the flight graph')
    graph = station_graph(city, node)
    _, labels = connected_components(graph, directed=False)
    connected = labels == labels[survey.initial[0]]
    required = np.r_[survey.initial, survey.waypoints[:, :ns].ravel()]
    if connected[node] or not connected[required].all():
        raise ValueError('Fixed relay station blocks a launch point or a required search waypoint')
    return graph, connected


def install_fixed(city, fleet, survey, node):
    city.graph, connected = mobile_component(city, survey, fleet.ns, node)
    city.distance, city.predecessor = shortest_path(city.graph, directed=False, return_predecessors=True)
    city.distance = city.distance.astype(np.float32)
    city.neighbors = np.full_like(city.neighbors, -1)
    for start in range(len(city.points)):
        end = city.graph.indices[city.graph.indptr[start]:city.graph.indptr[start + 1]]
        city.neighbors[start, 0] = start
        city.neighbors[start, 1:len(end) + 1] = end

    options = survey.options.copy()
    options[~connected[np.maximum(options, 0)]] = -1
    targets = survey.waypoints.copy()
    for stage, goals in enumerate(targets):
        for agent, original in enumerate(goals[fleet.ns:], fleet.ns):
            if connected[original]:
                continue
            alternatives = options[stage, agent - fleet.ns].ravel()
            available = next((candidate for candidate in alternatives if candidate >= 0 and candidate not in goals), None)
            if available is None:
                raise ValueError('Station would remove all safe targets for a moving relay')
            goals[agent] = available
    stationary = np.full((len(targets), 1, *options.shape[2:]), -1, dtype=options.dtype)
    stationary[..., 0] = node
    changed = replace(survey, initial=np.r_[survey.initial, np.int32(node)],
                      waypoints=np.column_stack((targets, np.full(len(targets), node, dtype=targets.dtype))),
                      options=np.concatenate((options, stationary), axis=1))
    city.radio_common_agents = fleet.n
    added = replace(fleet, n=fleet.n + 1, key=f'{fleet.key}_plus_fixed')
    return added, changed


class LayoutFlight(MainFlight):
    def __init__(self, model, threshold, layout):
        super().__init__(model, threshold)
        self.layout = layout

    def survey(self, *args):
        city, scenario, fleet, survey, anchor = super().survey(*args)
        city = copy(city)
        layout = self.layout
        if layout.fixed_node is not None:
            fleet, survey = install_fixed(city, fleet, survey, layout.fixed_node)
        self.altitudes = np.full(fleet.n, scenario.altitude_m, dtype=float)
        if layout.elevated_relay is not None:
            n_mobile = fleet.n - (layout.fixed_node is not None)
            if (not fleet.ns <= layout.elevated_relay < n_mobile or layout.height_m is None
                    or not np.isfinite(layout.height_m) or layout.height_m < scenario.altitude_m):
                raise ValueError('Elevated role must be a moving relay at or above the planned flight height')
            self.altitudes[layout.elevated_relay] = layout.height_m
            city.radio_altitudes_m = self.altitudes
        elif layout.height_m is not None:
            raise ValueError('Elevation needs a designated relay')
        self.history.bind(city, fleet)
        return city, scenario, fleet, survey, anchor


def run(spec, layout, max_seconds=MAX_MISSION):
    with map_scope(spec['region']):
        flight = LayoutFlight(*load_model(), layout)
        row, arrays = flight.run(spec, max_seconds)
    row.update(controller='halfsec_radio_layout', layout=asdict(layout), altitudes_m=flight.altitudes.tolist())
    if layout.fixed_node is not None:
        position = arrays['positions'][:, -1]
        np.testing.assert_array_equal(position, np.broadcast_to(position[0], position.shape))
        row['fixed_position_m'] = position[0].tolist()
    return row, arrays
