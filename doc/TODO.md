# TODO

Open questions about `calculate-loq` that aren't tracked as GitHub issues yet.

## Decide whether the `notes` column stays in figuresofmerit.csv

Added by `2205392` ("Never drop a peptide: record fit failures as noted rows"), which
changed the output schema from

```
peptide,LOD,LOQ,ULOQ,slope_linear,intercept_linear,intercept_noise,stndev_noise
```

to that plus `n_curvepoints` and `notes`.

The question is whether `notes` earns its place in the output, or whether the
never-drop-a-peptide guarantee is better expressed some other way (a sentinel in an
existing column, a separate sidecar file, a log line). Worth deciding before the
schema is baked into published supplemental tables.

Things to weigh:

- **What actually goes in it.** On the two EncyclopeDIA datasets re-run on 2026-08-12
  (Exploris 62,760 peptides; IL15 539) the column was empty for every single row. If
  it is usually empty, is a column the right shape for it?
- **It is a free-text column in a numeric table.** Anything consuming the CSV has to
  cope with a string field that may contain commas/quotes. Consider a small
  closed vocabulary of codes if it stays.
- **Downstream readers.** `2026_calcurve_paper` reads these CSVs with
  `pandas.read_csv` by column name, so extra columns are harmless there — but the
  manuscript ships these files as supplemental tables, where a column that is empty
  in every row invites reviewer questions.
- **`n_curvepoints` is a separate call.** It arrived in the same schema change but is
  a populated numeric field, so it stands on its own merits.

Raised while regenerating the paper's figures of merit; see
`2026_calcurve_paper/docs/fom_provenance.md` for the run that surfaced it.
