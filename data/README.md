# Data layout and access policy

The raw SMR flowsheet data are not distributed in this Git repository. They contain simulator-derived process information and are managed separately under the project data-access policy.

To reproduce training or evaluation, place approved local copies at the paths expected by the configurations:

```text
data/
├── datasets_v3/
│   └── process_main_merged.csv
├── main_data_Streams/
│   └── process_1_10_stream_edge_mapping_final.csv
├── process_specs/raw/
│   └── Process*_Adjacency_Matrix.xlsx
├── reference/v3/
│   ├── canonical graph-spec files
│   ├── HX edge-pair metadata
│   └── excluded-sample manifest
└── splits/
    ├── all_processes_full100k_outer5_grouped_60_20_20/
    └── single_process_full_unseen_60_20_20/
```

The exact paths used by the final protocol are declared in `configs/final_experiments/final_protocol_260811.yaml` and `configs/experiment/pinn/model_260805_10d_frac1.yaml`.

Do not replace a canonical split manifest with a newly shuffled split when reproducing a reported result. Preprocessing and scaling must be fit with training rows only.

For a public release, publish only a de-identified and approved dataset version through an archival repository, then add its DOI and retrieval instructions to the root README.
