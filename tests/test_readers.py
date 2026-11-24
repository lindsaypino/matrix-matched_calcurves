"""Round-trip tests for the input readers that have no other test.

Each test re-expresses data/one_protein.csv (EncyclopeDIA, wide) in one search engine's
output format, reads it back with read_input, and checks that the result is the frame
the EncyclopeDIA file gives, with the peptide names that format produces. A reader that
detects the wrong format, drops or renames a run, or mangles a peptide name fails here.

The DIA-NN diann_report.tsv reader is covered by the issue #15 and #19 tests in
test_issues.py.
"""

import importlib
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "bin"))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CURVE_DATA = os.path.join(REPO, "data", "one_protein.csv")
CONC_MAP = os.path.join(REPO, "data", "filename2samplegroup_map.csv")

COLS = ["peptide", "curvepoint", "area", "n_detected"]
CHARGE = 2


@pytest.fixture
def calc():
    return importlib.import_module("calculate-loq")


def _wide():
    """one_protein.csv as peptide x run, without the EncyclopeDIA-only columns."""
    return pd.read_csv(CURVE_DATA).drop(columns=["Protein", "numFragments"])


def _expected(calc, rename):
    """The frame the EncyclopeDIA file gives, with peptides renamed the way the
    format under test names them, in canonical order."""
    df = calc.read_input(CURVE_DATA, CONC_MAP)[COLS]
    df["peptide"] = df["peptide"].map(rename)
    return (df.sort_values(["peptide"] + calc.SORT_KEYS, kind="mergesort")
              .reset_index(drop=True))


def _read(calc, path, capsys):
    df = calc.read_input(str(path), CONC_MAP)[COLS]
    return df, capsys.readouterr().out


def test_skyline_export_reads_back_to_the_wide_frame(calc, tmp_path, capsys):
    """Skyline export: long format, comma-separated, one row per peptide per run."""
    long = _wide().melt(id_vars="Peptide", var_name="File Name",
                        value_name="Total Area Fragment")
    long = long.rename(columns={"Peptide": "Peptide Sequence"})
    long.insert(0, "Protein Name", "PMA1_YEAST")  # an extra column the reader must ignore
    path = tmp_path / "skyline_export.csv"
    long.to_csv(path, index=False)

    df, out = _read(calc, path, capsys)

    assert "Skyline" in out
    pd.testing.assert_frame_equal(df, _expected(calc, lambda p: p), check_exact=True)


def test_diann_pr_matrix_reads_back_to_the_wide_frame(calc, tmp_path, capsys):
    """DIA-NN *.pr_matrix.tsv: wide, tab-separated. The peptide is
    Modified.Sequence + "_" + charge, with the colons of UniMod tags removed."""
    wide = _wide()
    peptides = list(wide["Peptide"])
    # give one peptide a UniMod tag, to check that its colon is removed
    tagged = peptides[0]
    modseq = {p: (p + "(UniMod:35)" if p == tagged else p) for p in peptides}

    meta = pd.DataFrame({
        "Protein.Group": "P05030",
        "Protein.Ids": "P05030",
        "Protein.Names": "PMA1_YEAST",
        "Genes": "PMA1",
        "First.Protein.Description": "Plasma membrane ATPase 1",
        "Proteotypic": 1,
        "Stripped.Sequence": peptides,
        "Modified.Sequence": [modseq[p] for p in peptides],
        "Precursor.Charge": CHARGE,
        "Precursor.Id": ["%s%d" % (modseq[p], CHARGE) for p in peptides],
    })
    matrix = pd.concat([meta, wide.drop(columns="Peptide")], axis=1)
    path = tmp_path / "report.pr_matrix.tsv"
    matrix.to_csv(path, sep="\t", index=False)

    df, out = _read(calc, path, capsys)

    assert "pr_matrix" in out
    expected = _expected(calc, lambda p: "%s_%d" % (modseq[p].replace(":", ""), CHARGE))
    assert "%s(UniMod35)_%d" % (tagged, CHARGE) in set(df["peptide"])
    pd.testing.assert_frame_equal(df, expected, check_exact=True)


@pytest.mark.parametrize("cam_tag", ["[Carbamidomethyl (C)]", "[+57.0214635]"],
                         ids=["named_mod", "mass_mod"])
def test_spectronaut_output_reads_back_to_the_wide_frame(calc, tmp_path, capsys, cam_tag):
    """Spectronaut: wide, tab-separated. The peptide comes from EG.PrecursorId
    ("_SEQUENCE_.charge") as SEQUENCE_charge.

    Spectronaut normally writes modifications by name (C[Carbamidomethyl (C)]). The
    mass form C[+57.0214635] has a dot of its own, which the reader once split on."""
    wide = _wide()
    peptides = list(wide["Peptide"])
    modseq = {p: p.replace("[+57.0214635]", cam_tag) for p in peptides}
    assert any("[" in modseq[p] for p in peptides), "no modified peptide; the test checks less"
    meta = pd.DataFrame({
        "PG.ProteinGroups": "P05030",
        "PG.Organisms": "Saccharomyces cerevisiae",
        "PG.ProteinNames": "PMA1_YEAST",
        "PEP.StrippedSequence": [p.replace("[+57.0214635]", "") for p in peptides],
        "PEP.PeptidePosition": 1,
        "EG.PrecursorId": ["_%s_.%d" % (modseq[p], CHARGE) for p in peptides],
        "EG.ModifiedSequence": ["_%s_" % modseq[p] for p in peptides],
    })
    table = pd.concat([meta, wide.drop(columns="Peptide")], axis=1)
    path = tmp_path / "spectronaut_report.tsv"
    table.to_csv(path, sep="\t", index=False)

    df, out = _read(calc, path, capsys)

    assert "Spectronaut" in out
    pd.testing.assert_frame_equal(df, _expected(calc, lambda p: "%s_%d" % (modseq[p], CHARGE)),
                                  check_exact=True)
