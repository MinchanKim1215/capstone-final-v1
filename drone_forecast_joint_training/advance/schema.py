"""관측·통계·지도 입력의 순서와 의미를 기계 판독 명세로 제공."""

from swarm_sim.regions import ROOT, read_json, write_json
from usable.spatial import NAMES as SPATIAL_NAMES
from advance.features import FEATURES, H, VERSION
from advance.data import RESULT


def main():
    manifest = read_json(ROOT / "datasets/usable45/collection_manifest.json")
    raw = manifest["records"][0]["metadata"]["feature_names"]
    names = raw + SPATIAL_NAMES
    history = [names[i] for i in FEATURES]
    statistics = []
    for prefix in [
        "latest_value",
        "latest_observed",
        "last5_mean",
        "last5_std",
        "last5_min",
        "last5_max",
        "last5_change",
        "last20_mean",
        "last20_std",
        "last20_min",
        "last20_max",
        "last20_change",
    ]:
        statistics += [prefix + "." + n for n in history]

    def geometry(prefix):
        return (
            [prefix + f".loss_change_{h:g}s_db" for h in H]
            + [prefix + f".los_{h:g}s" for h in H]
            + [prefix + f".margin_{h:g}s_db" for h in H]
            + [
                prefix + "." + n
                for n in [
                    "max_loss_first5",
                    "min_margin_first5",
                    "max_loss_lead3to4",
                    "min_margin_lead3to4",
                    "max_loss_lead3to7",
                    "min_margin_lead3to7",
                    "supported",
                ]
            ]
        )

    physics = (
        [f"physics.residual_margin_{h:g}s_db" for h in H]
        + [f"physics.gaussian_marginal_risk_{h:g}s" for h in H]
        + [
            "physics." + n
            for n in [
                "shadow_residual",
                "inbound_noise_interference",
                "reverse_noise_proxy",
                "mean_noise_5s",
                "mean_reverse_noise_5s",
                "noise_change_5s",
                "reverse_noise_change_5s",
                "shadow_change_5s",
                "max_risk_first5",
                "onset_proxy3to4",
            ]
        ]
    )
    goal = [
        "self_goal_dx",
        "self_goal_dy",
        "received_peer_goal_dx",
        "received_peer_goal_dy",
        "peer_command_age_s",
        "seconds_to_control_boundary",
        "self_goal_distance",
        "peer_goal_distance",
        "command_supported",
    ]
    extra = geometry("cv") + physics + geometry("goal") + ["intent." + n for n in goal]
    assert len(history) == 55 and len(statistics) == 660 and len(extra) == 137
    payload = {
        "feature_version": VERSION,
        "raw_observation_fields": raw,
        "raw_shape": "[N,N,48] or local [N,48]",
        "mask_rule": "missing -> zero and False",
        "history_input_fields": history,
        "history_half_second_steps": 40,
        "statistics_fields": statistics,
        "extra_fields": extra,
        "goal_input_fields": goal,
        "goal_input_shape": "[N,N,9] or local [N,9]",
        "forbidden_inputs": [
            "actual future positions",
            "unissued future commands",
            "future jammer position/power/timing",
            "latent true SINR",
            "true current link availability",
        ],
        "beacon_budget_bytes": {
            "header": 16,
            "position_xyz": 12,
            "velocity_xyz": 12,
            "current_goal_xy_float64": 16,
            "issue_time_uint32": 4,
            "seven_reverse_reports": 49,
            "crc32": 4,
            "padding": 15,
        },
        "abstract_serialization_budget": True,
        "prediction_outputs": {
            "risk_5s": "future >=1s unavailability inside5s",
            "risk_onset_3to4s": "new persistent outage onset3to4s ahead",
            "risk_onset_3to7s": "new persistent outage onset3to7s ahead",
            "unsupported_probability": None,
        },
        "role_assignment_fixed": {"total": 8, "search": 3, "relay": 5},
    }
    assert sum(payload["beacon_budget_bytes"].values()) == 128
    write_json(RESULT / "input_output_schema.json", payload)


if __name__ == "__main__":
    main()
