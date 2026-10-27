from pathlib import Path

import gemmi
import pandas as pd

from gsbind.pinder_pipeline import (
    MANIFEST_COLUMNS,
    PipelineOptions,
    PinderApiAdapter,
    add_scores,
    anchor_selection,
    enforce_cluster_cap,
    quality_mask,
    run_pipeline,
    select_candidate_and_test_systems,
    sequence_diverse_select,
    union_receptor_holos,
)


def _index() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": "train-clean", "split": "train", "cluster_id_R": "cluster_clean",
            "cluster_id_L": "cluster_other", "uniprot_R": "P01", "uniprot_L": "P02",
            "holo_R_pdb": "train-clean-R.pdb", "holo_L_pdb": "train-clean-L.pdb",
        },
        {
            "id": "val-clean", "split": "val", "cluster_id_R": "cluster_val",
            "cluster_id_L": "cluster_val_partner", "uniprot_R": "P03", "uniprot_L": "P04",
            "holo_R_pdb": "val-clean-R.pdb", "holo_L_pdb": "val-clean-L.pdb",
        },
        {
            "id": "train-leak", "split": "train", "cluster_id_R": "cluster_test_R",
            "cluster_id_L": "cluster_other", "uniprot_R": "P05", "uniprot_L": "P06",
            "holo_R_pdb": "train-leak-R.pdb", "holo_L_pdb": "train-leak-L.pdb",
        },
        {
            "id": "test-case", "split": "test", "cluster_id_R": "cluster_test_R",
            "cluster_id_L": "cluster_test_L", "uniprot_R": "P07", "uniprot_L": "P08",
            "holo_R_pdb": "test-case-R.pdb", "holo_L_pdb": "test-case-L.pdb",
        },
    ])


def test_split_selection_excludes_both_official_test_side_clusters():
    training, test, audit = select_candidate_and_test_systems(_index())
    assert set(training["id"]) == {"train-clean", "val-clean"}
    assert set(test["id"]) == {"test-case"}
    assert audit["official_test_side_clusters"] == 2


def test_quality_contract_recomputes_positive_rate_and_ignores_stale_value():
    options = PipelineOptions(output_dir=Path("unused"))
    frame = pd.DataFrame([
        {"num_residues": 100, "num_positive_labels": 75, "mask_coverage": 1.0,
         "low_mapping_quality": 0, "coordinate_union_status": "ok", "sequence_identity": "a"},
        {"num_residues": 100, "num_positive_labels": 76, "mask_coverage": 1.0,
         "low_mapping_quality": 0, "coordinate_union_status": "ok", "sequence_identity": "b"},
        {"num_residues": 100, "num_positive_labels": 76, "mask_coverage": 1.0,
         "low_mapping_quality": 0, "coordinate_union_status": "error", "sequence_identity": "c"},
    ])
    assert list(quality_mask(frame, options)) == [True, False, False]


def test_union_receptor_holos_or_labels_and_counts_events():
    frame = pd.DataFrame([
        {"receptor_holo_pdb": "h", "sequence": "AAA", "cluster_id": "c", "system_id": "a",
         "receptor_side": "R", "partner_holo_pdb": "p1", "labels": "1;0;0",
         "input_residue_ids": "1;2;3", "num_residues": 3, "mask_coverage": 1.0,
         "low_mapping_quality": 0, "coordinate_union_status": "ok", "sequence_identity": "s"},
        {"receptor_holo_pdb": "h", "sequence": "AAA", "cluster_id": "c", "system_id": "b",
         "receptor_side": "R", "partner_holo_pdb": "p2", "labels": "0;1;0",
         "input_residue_ids": "1;2;3", "num_residues": 3, "mask_coverage": 1.0,
         "low_mapping_quality": 0, "coordinate_union_status": "ok", "sequence_identity": "s"},
    ])
    union = union_receptor_holos(frame)
    assert len(union) == 1
    assert union.iloc[0]["labels"] == "1;1;0"
    assert union.iloc[0]["holo_event_count"] == 2
    assert union.iloc[0]["positive_rate"] == 2 / 3


def test_sequence_diverse_cluster_cap_uses_unique_sequences_and_score_order():
    frame = pd.DataFrame([
        {"receptor_holo_pdb": "best", "receptor_side": "R", "cluster_id": "cluster_a",
         "sequence_identity": "same", "num_residues": 100, "mask_coverage": 1.0,
         "holo_event_failure_count": 0, "holo_event_success_count": 2},
        {"receptor_holo_pdb": "duplicate", "receptor_side": "R", "cluster_id": "cluster_a",
         "sequence_identity": "same", "num_residues": 100, "mask_coverage": 1.0,
         "holo_event_failure_count": 0, "holo_event_success_count": 1},
        {"receptor_holo_pdb": "different", "receptor_side": "R", "cluster_id": "cluster_a",
         "sequence_identity": "different", "num_residues": 100, "mask_coverage": 1.0,
         "holo_event_failure_count": 0, "holo_event_success_count": 1},
    ])
    scored = add_scores(frame)
    selected = sequence_diverse_select(scored, cap=5)
    assert selected == {"best": 1, "different": 2}


def test_anchor_selection_prioritizes_historical_origins(tmp_path):
    frame = pd.DataFrame([
        {"receptor_holo_pdb": "old", "receptor_side": "R", "cluster_id": "c", "sequence_identity": "s1"},
        {"receptor_holo_pdb": "new", "receptor_side": "R", "cluster_id": "c", "sequence_identity": "s2"},
    ])
    anchor = tmp_path / "anchor.csv"
    anchor.write_text(
        "holo_pdb,cluster_id,selection_origin,sequence_diversity_slot\n"
        "old,c,relaxed10477_strict_official_cluster_recollapse_v1,9\n"
        "new,c,invalid_official_val_side_overlap,1\n",
        encoding="utf-8",
    )
    selected, audit = anchor_selection(frame, anchor, cap=1)
    assert selected == {"old": 1}
    assert audit["anchor_rows_dropped_by_global_cap"] == 1


def _write_native_pdb(path: Path, system_id: str) -> None:
    structure = gemmi.Structure()
    structure.name = system_id
    model = gemmi.Model("1")
    for chain_name in ("R", "L"):
        chain = gemmi.Chain(chain_name)
        for index in range(50):
            residue = gemmi.Residue()
            residue.name = "ALA"
            residue.seqid = gemmi.SeqId(index + 1, " ")
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            x = 0.0 if chain_name == "R" else (4.0 if index < 10 else 100.0)
            atom.pos = gemmi.Position(x, 0.0, float(index * 10))
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(path))


class FakeAdapter:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def native_path(self, system_id: str) -> Path:
        return self.cache_dir / "pdbs" / f"{system_id}.pdb"

    def download_native(self, system_ids) -> None:
        for system_id in system_ids:
            _write_native_pdb(self.native_path(system_id), system_id)

    def describe(self) -> dict[str, object]:
        return {"package": "synthetic-test", "release": "2024-02"}


def test_end_to_end_rebuild_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "pinder-cache"
    index = _index()
    index.loc[index["id"].eq("train-clean"), "cluster_id_R"] = "cluster_train_unique"
    adapter = FakeAdapter(cache)
    options = PipelineOptions(output_dir=tmp_path / "rebuild", workers=1, chunk_size=4)
    report = run_pipeline(index, adapter, options)
    train = pd.read_csv(tmp_path / "rebuild/train.csv")
    test = pd.read_csv(tmp_path / "rebuild/test.csv")
    assert list(train.columns) == MANIFEST_COLUMNS
    assert len(train) == 4
    assert len(test) == 2
    assert all(row["labels"].count("1") == 10 for _, row in train.iterrows())
    assert report["checks"]["train_test_case_overlap"] == 0
    assert (tmp_path / "rebuild/split_contract.json").is_file()
    assert report["checks"]["max_train_cluster_size"] == 1
