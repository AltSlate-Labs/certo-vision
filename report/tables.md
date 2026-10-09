## Trial table

| Question | type | n | ≈ calib images per option (smallest) | zero-shot | calibrated | few-shot | majority | ECE few-shot | NLL zero, few-shot | MAE / ρ | in-scope abstain | OOD rejected (n) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bottle: defective? | noul | 183 | 19 | 0.640 | 0.874 | **0.932** ± 0.011 | 0.640 | 0.056 | 0.63, 0.18 |  | 0.10 | 100% (659) |
| hazelnut: defective? | noul | 210 | 21 | 0.751 | 0.950 | **0.951** ± 0.006 | 0.658 | 0.033 | 0.53, 0.13 |  | 0.06 | 100% (639) |
| carpet: defective? | noul | 217 | 27 | 0.600 | 0.641 | **0.940** ± 0.019 | 0.578 | 0.051 | 0.71, 0.16 |  | 0.05 | 100% (651) |
| screw: defective? | noul | 260 | 36 | 0.537 | 0.617 | **0.712** ± 0.024 | 0.537 | 0.095 | 0.88, 0.55 |  | 0.04 | 100% (638) |
| hazelnut: defect type (4) | choice | 70 | 5 | 0.490 | 0.837 | **0.869** ± 0.027 | 0.274 | 0.098 | 1.05, 0.38 |  | 0.10 | 100% (639) |
| screw: defect type (5) | choice | 119 | 7 | 0.243 | 0.200 | **0.364** ± 0.054 | 0.238 | 0.131 | 1.60, 1.47 |  | 0.02 | 100% (638) |
| banana: ripeness (score, 4) | score | 600 | 45 | 0.483 | 0.626 | **0.833** ± 0.025 | 0.267 | 0.044 | 1.09, 0.44 | 0.23 / 0.90 | 0.09 | 100% (370) |
| banana: overripe? | noul | 600 | 45 | 0.799 | 0.801 | **0.929** ± 0.014 | 0.745 | 0.028 | 0.48, 0.17 |  | 0.09 | 100% (370) |
| banana: stage (4) | choice | 600 | 45 | 0.499 | 0.600 | **0.785** ± 0.028 | 0.267 | 0.051 | 1.16, 0.58 |  | 0.09 | 100% (370) |

## STL-10 table

| Backbone | task | zero-shot | calibrated | few-shot | majority | ECE few-shot | embed ms (GB10) |
|---|---|---|---|---|---|---|---|
| SigLIP 2 so400m | subject (10-way) | 0.990 | 0.993 | **0.995** ± 0.001 | 0.110 | 0.005 | 72 |
| SigLIP 2 so400m | is it a cat? | 0.900 | 0.900 | **0.990** ± 0.005 | 0.900 | 0.016 | 72 |
| SigLIP 2 so400m | blur level (5, synthetic) | 0.247 | 0.340 | **0.878** ± 0.004 | 0.207 | 0.028 | 72 |
| EmbeddingGemma 2 | subject (10-way) | 0.967 | 0.983 | **0.988** ± 0.002 | 0.110 | 0.011 | 84 |
| EmbeddingGemma 2 | is it a cat? | 0.126 | 0.900 | **0.964** ± 0.021 | 0.900 | 0.033 | 84 |
| EmbeddingGemma 2 | blur level (5, synthetic) | 0.232 | 0.202 | **0.857** ± 0.015 | 0.207 | 0.043 | 84 |

## Scope table

| Backbone | question | out-of-scope set | AUROC NN ratio | AUROC max-prob | AUROC MRL agreement | rejected by NN gate | max-prob > 0.9 |
|---|---|---|---|---|---|---|---|
| SigLIP 2 | choice | textures (DTD) | 1.000 | 0.999 | 0.722 | 100% | 1% |
| SigLIP 2 | choice | satellite (EuroSAT) | 0.992 | 1.000 | 0.669 | 100% | 0% |
| SigLIP 2 | choice | Oxford pets | 0.979 | 0.567 | 0.352 | 93% | 100% |
| SigLIP 2 | noul | textures (DTD) | 1.000 | 0.680 | 0.625 | 100% | 97% |
| SigLIP 2 | noul | satellite (EuroSAT) | 0.992 | 0.484 | 0.600 | 100% | 100% |
| SigLIP 2 | noul | Oxford pets | 0.979 | 0.837 | 0.882 | 93% | 79% |
| EmbeddingGemma 2 | choice | textures (DTD) | 1.000 | 0.993 | 0.923 | 100% | 2% |
| EmbeddingGemma 2 | choice | satellite (EuroSAT) | 1.000 | 0.926 | 0.917 | 100% | 40% |
| EmbeddingGemma 2 | choice | Oxford pets | 0.996 | 0.731 | 0.625 | 98% | 100% |
| EmbeddingGemma 2 | noul | textures (DTD) | 1.000 | 0.424 | 0.408 | 100% | 96% |
| EmbeddingGemma 2 | noul | satellite (EuroSAT) | 1.000 | 0.356 | 0.371 | 100% | 100% |
| EmbeddingGemma 2 | noul | Oxford pets | 0.996 | 0.729 | 0.744 | 98% | 58% |
