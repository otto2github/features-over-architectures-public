# PeerJ CS revision experiments (v1.1)

Round 1 addressed reviewer requests for an imbalance-specialized fraud GNN, a
reproducible score-persisted headline run, and a coarse hyperparameter sweep.
Round 2 addressed three blocking concerns: audit-opinion ablation, multi-seed
pure-graph diagnostic with non-GNN structural baselines, and code-path reconciliation.

All scripts use: scripts/training/gnn_baseline_common.py and gnn_baseline_v1_1.py.
Set THESIS_PROJECT_ROOT to the project root before running.

## Scripts

### Round 1
rerun_main_persist_scores.py  -- MLP + 4 GNNs x5 seeds, persists per-firm-year scores  (Sec V-C, Table 5c)
run_pcgnn.py                  -- PC-GNN-style variant x5 seeds                          (Sec V-C)
gnn_tuning_sweep.py           -- lr x hidden coarse sweep, seed=42, M11                 (Sec V-C, Table 5d)

### Round 2
exp1_audit_ablation.py        -- Five-seed audit-opinion ablation (M5/M11-audit/M11 x MLP+4GNNs)   (Sec V-H, Table 10)
exp2_puregraph_structural.py  -- Pure-graph multiseed + Node2Vec->LR + label propagation baselines  (Sec V-D, Tables 6,6b)
exp3_reconciliation.py        -- Code-path reconciliation, reads existing result JSONs              (Sec V-C, Table 5e)
exp4_tabular_baselines.py     -- Tabular baselines (unweighted + class-weighted) x5 seeds          (Sec V-A, Tables 3,3b)

Run example (from scripts/training/):
  THESIS_PROJECT_ROOT=/path/to/project python rerun_main_persist_scores.py --modal M11 --label-col fraud_v08_strict --lr 5e-4
  THESIS_PROJECT_ROOT=/path/to/project python exp2_puregraph_structural.py --label-col fraud_v08_strict

## Result artifacts

### Round 1
rerun_persist_fraud_v08_strict_M11.json          -- Score-persisted rerun, MLP 0.7190+/-0.0022 highest
rerun_persist_fraud_v08_strict_M11_delong.json   -- Pooled five-seed-mean-score DeLong
pcgnn_fraud_v08_strict_M11.json                  -- PC-GNN-style variant, 0.7126+/-0.0039
tuning_sweep_fraud_v08_strict_M11_seed42.{json,csv}
scores_persist/{MODEL}_M11_fraud_v08_strict_{seed}_scores.npz  -- y_score/y_true, n=19384

### Round 2
puregraph_structural_fraud_v08_strict_20260606_152833.json
  Output of exp2. Keys: pure_graph_multiseed (GCN 0.5742+/-0.0008, GAT 0.5067+/-0.0008,
  GraphSAGE 0.4910+/-0.0368, RGCN 0.5874+/-0.0049); node2vec_lr 0.6257 (n_test 19384);
  label_propagation 0.5329 (n_test 19384). Populates Tables 6 and 6b.
  File archived at ../data/results/ and in this directory.

Audit-ablation result: ../data/results/audit_ablation_fraud_v08_strict_20260606_134048.json
  Populates Table 10. Run exp1 to regenerate.

Tabular and reconciliation results: reproduced by running exp4 and exp3 against
the features/edges/labels in the project data directory.

## Graph note (entity-level)
node_id_index: 303771 nodes -- 5337 listed companies + 209000 persons + 89434 institutions
global_edge_index: 4469735 directed edges, types E1-E5, per-year attribute
Each year-t graph is a SINGLE SHARED graph: all companies listed in year t carry
their features simultaneously. GNNs receive feature-bearing company-to-company paths.
Person/institution nodes and companies not listed in year t are zero-initialized.
51675 firm-year observations (train 24527 / val 7764 / test 19384).
No temporal edges. Raw CSMAR data not redistributed (licensed).
