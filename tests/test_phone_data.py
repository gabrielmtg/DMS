import csv
import json

import numpy as np
import pytest

from dms_train.phone_data import (
    class_weights,
    confusion_matrix,
    label_mapping,
    load_config,
    load_or_make_splits,
    macro_f1,
    make_splits,
    per_class_scores,
    read_manifest,
    select_rows,
)

CFG = {
    "classes": {"normal": ["Safe Driving"], "phone_call": ["Phone call (Right)", "Phone call (Left)"]},
    "ignore": ["Texting (Right)"],
}
SPLIT_CFG = {"seed": 0, "fractions": {"val": 0.19, "test": 0.16}}


def test_mapping_merges_sides_and_ignores_texting():
    mapping = label_mapping(CFG)
    assert mapping["Phone call (Left)"] == mapping["Phone call (Right)"] == "phone_call"
    assert mapping["Texting (Right)"] is None


def test_repeated_label_is_an_error():
    cfg = {"classes": {"a": ["X"], "b": ["X"]}}
    with pytest.raises(ValueError, match="repetido"):
        label_mapping(cfg)


def test_project_config_has_four_classes_and_no_repeats():
    cfg = load_config("configs/phone.yaml")
    assert list(cfg["classes"]) == ["normal", "phone_call", "drinking", "hair_makeup"]
    label_mapping(cfg)


def test_select_rows_samples_frames_and_drops_ignored():
    rows = [{"label": "Safe Driving", "frame": f} for f in range(25)]
    rows += [{"label": "Texting (Right)", "frame": 30}]
    selected = select_rows(rows, label_mapping(CFG), every_n_frames=10)
    assert [row["frame"] for row in selected] == [0, 10, 20]
    assert {row["class"] for row in selected} == {"normal"}


def test_unknown_labels_are_listed_in_the_error():
    rows = [{"label": "safe_drive", "frame": 0}, {"label": "safe_drive", "frame": 10}]
    with pytest.raises(ValueError, match=r"'safe_drive' \(2\)"):
        select_rows(rows, label_mapping(CFG), every_n_frames=10)


def test_manifest_requires_columns(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("video,frame,label\nv.mp4,0,Safe Driving\n")
    with pytest.raises(ValueError, match="subject"):
        read_manifest(path)


def test_manifest_reads_frame_as_int(tmp_path):
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([["video", "frame", "label", "subject"], ["v.mp4", "12", "Safe Driving", "1"]])
    assert read_manifest(path)[0]["frame"] == 12


def test_splits_by_subject_are_disjoint_and_match_dmd_paper():
    splits = make_splits([f"s{i}" for i in range(37)], SPLIT_CFG["fractions"], seed=0)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [24, 7, 6]
    assert len(set(splits["train"]) | set(splits["val"]) | set(splits["test"])) == 37
    assert splits == make_splits([f"s{i}" for i in range(37)], SPLIT_CFG["fractions"], seed=0)


def test_splits_need_enough_subjects():
    with pytest.raises(ValueError, match="insuficientes"):
        make_splits(["a", "b"], SPLIT_CFG["fractions"], seed=0)


def test_saved_split_is_reused_and_rejects_new_subjects(tmp_path):
    path = tmp_path / "splits.json"
    first = load_or_make_splits(path, ["a", "b", "c", "d"], SPLIT_CFG)
    assert json.loads(path.read_text()) == first
    assert load_or_make_splits(path, ["a", "b"], SPLIT_CFG) == first
    with pytest.raises(ValueError, match="'e'"):
        load_or_make_splits(path, ["a", "e"], SPLIT_CFG)


def test_scores_on_a_known_confusion_matrix():
    cm = confusion_matrix([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 0, 2], 3)
    assert cm.tolist() == [[2, 1, 0], [1, 1, 0], [0, 0, 1]]
    scores = per_class_scores(cm)
    assert np.allclose(scores["precision"], [2 / 3, 1 / 2, 1])
    assert np.allclose(scores["recall"], [2 / 3, 1 / 2, 1])
    assert macro_f1(cm) == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)


def test_macro_f1_ignores_classes_without_examples():
    cm = confusion_matrix([0, 1], [0, 1], 3)
    assert macro_f1(cm) == 1.0


def test_class_weights_balance_and_skip_missing_classes():
    counts = np.array([30, 10, 0])
    weights = class_weights(counts)
    assert weights[2] == 0
    assert weights[0] * 30 == pytest.approx(weights[1] * 10)
    assert (weights * counts).sum() / counts.sum() == pytest.approx(1.0)
