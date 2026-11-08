"""고정한 트리의 학습 분할 이득 요약. 성능 평가·재선정과 분리."""

from collections import defaultdict
import numpy as np
from swarm_sim.regions import ROOT, read_json, write_json
from advance.data import RESULT
from advance.features import FEATURES
from advance.infer import selected, load


def category(q):
    if q < 660:
        field = FEATURES[q % 55]
        if field >= 48:
            return "정적 지도 관측 이력"
        if field in [0, 1, 2, 3, 4, 11, 27, 47]:
            return "자기 이동·역할·상태 이력"
        if field in [15, 16, 17, 18, 19, 20, 21, 34, 35, 36, 37, 38, 39]:
            return "이웃 위치·상대 이동 이력"
        return "통신 품질·패킷·수신 이력"
    e = q - 660
    if e < 46:
        return "현재 속도 유지의 지도 외삽"
    if e < 82:
        return "측정 잔차·시간 변화 보조"
    if e < 128:
        return "현재 목표 유지의 지도 외삽"
    return "자기 발행·수신 목표 상태"


def main():
    path, choice = selected()
    model, state = load(path)
    if state["architecture"] != "gbdt":
        raise ValueError("트리 선택의 분할 이득만 지원")
    schema = read_json(RESULT / "input_output_schema.json")
    names = schema["statistics_fields"] + schema["extra_fields"]
    assert len(names) == 797
    heads = {}
    for head in [0, 3]:
        gain = model.trees[head].feature_importance(importance_type="gain")
        total = float(gain.sum())
        parts = defaultdict(float)
        for q, value in enumerate(gain):
            parts[category(q)] += float(value) / max(total, 1e-12)
        top = np.argsort(gain)[-12:][::-1]
        heads[str(head)] = {
            "category_gain_fraction": dict(parts),
            "top_features": [
                {"feature": names[q], "gain_fraction": float(gain[q]) / max(total, 1e-12)}
                for q in top
            ],
        }
    out = {
        "checkpoint_sha256": choice["checkpoint_sha256"],
        "training_split_gain_only": True,
        "no_test_tuning": True,
        "not_causal_effect_or_ablation": True,
        "correlated_features_can_share_or_displace_gain": True,
        "heads": heads,
    }
    write_json(RESULT / "training_feature_gain.json", out)
    print(out)


if __name__ == "__main__":
    main()
