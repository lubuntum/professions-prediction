from datetime import datetime, timezone

from clusters import ClusterState, write_cluster_state
from mapping import load_active_features, select_active_features
from models import PsychParam, PsychTest, Pupil, Specialist
from prediction import predict_pupil
from specialist import create_cluster_files


def psych_tests(active_features, values):
    grouped = {}
    for feature_name, value in values.items():
        test_name = active_features.test_by_feature[feature_name]
        grouped.setdefault(test_name, []).append(PsychParam(name=feature_name, value=value))
    return {
        test_name: PsychTest(test_type_name=test_name, psych_params=params)
        for test_name, params in grouped.items()
    }


def test_mapping_uses_only_tests_with_is_active_true(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        """{
          "test_mapping": {
            "A": {"param_names": ["active_value"], "is_active": true},
            "B": {"param_names": ["inactive_value"], "is_active": false}
          },
          "default_value": 0
        }""",
        encoding="utf-8",
    )
    active_features = load_active_features(mapping_path)
    tests = {
        "A": PsychTest(
            test_type_name="A",
            psych_params=[PsychParam(name="active_value", value=10)],
        ),
        "B": PsychTest(
            test_type_name="B",
            psych_params=[PsychParam(name="inactive_value", value=999)],
        ),
    }

    assert active_features.names == ("active_value",)
    assert select_active_features(tests, active_features) == [10.0]


def test_null_parameter_uses_mapping_default(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        """{
          "test_mapping": {
            "A": {"param_names": ["active_value"], "is_active": true}
          },
          "default_value": 3.5
        }""",
        encoding="utf-8",
    )
    active_features = load_active_features(mapping_path)
    tests = {
        "A": PsychTest(
            test_type_name="A",
            psych_params=[PsychParam(name="active_value", value=None)],
        )
    }

    assert select_active_features(tests, active_features) == [3.5]


def test_specialist_and_pupil_use_the_same_feature_order(settings):
    active_features = load_active_features(settings.mapping_path)
    values = {name: float(index + 1) for index, name in enumerate(active_features.names)}
    tests = psych_tests(active_features, values)
    specialist = Specialist(specialist_id=1, profession="A", psych_tests=tests)
    pupil = Pupil(pupil_id=2, psych_tests=tests)

    assert select_active_features(specialist.psych_tests, active_features) == list(values.values())
    assert select_active_features(pupil.psych_tests, active_features) == list(values.values())


def test_prediction_is_deterministic_on_fixed_specialists(settings):
    active_features = load_active_features(settings.mapping_path)
    first_feature = active_features.names[0]
    specialists = [
        Specialist(
            specialist_id=index + 1,
            profession="Profession A" if value < 5 else "Profession B",
            psych_tests=psych_tests(active_features, {first_feature: value}),
        )
        for index, value in enumerate((0.0, 1.0, 2.0, 8.0, 9.0, 10.0))
    ]
    cluster_folder = settings.cluster_dir / "fixture"
    build = create_cluster_files(
        specialists,
        cluster_folder,
        active_features,
        settings.cluster_count,
    )
    write_cluster_state(
        settings,
        ClusterState(datetime.now(timezone.utc), "fixture", build.specialists_count),
    )
    pupil = Pupil(
        pupil_id=100,
        psych_tests=psych_tests(active_features, {first_feature: 1.2}),
    )

    first = predict_pupil(pupil, settings)
    second = predict_pupil(pupil, settings)

    assert first == second
    assert first.pupil_id == 100
    assert first.predicted_profession == "Profession A"
    assert first.nearest_specialist_id == 2
    assert first.distance >= 0
