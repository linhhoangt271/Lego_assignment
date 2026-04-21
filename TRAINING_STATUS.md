# V3 Hyperparameter Tuning + Cross-Validation Pipeline

## ✅ Status: COMPLETE - ALL PHASES FINISHED

**Script**: `optionB_v3_hpopt.py` (1,051 lines)  
**Start Time**: Apr 8, ~15:24 UTC  
**Completion Time**: Apr 9, 20:34 UTC  
**Total Duration**: ~29 hours 10 minutes

---

## 🔍 Process Status

✅ **COMPLETE** - All processes finished successfully  
Final logs saved to: `optionB_v3_hpopt_training.log`

---

## 📊 Pipeline Progress

### ✅ Phase 1: Optuna Hyperparameter Search (COMPLETE)
**30 trials across 3 variants - COMPLETED on Apr 9, 11:10 UTC**

**Results Summary:**
| Loss Variant | Best Val F1 | Best LR | Best t0 |
|-------------|------------|---------|---------|
| **Focal**   | 0.6833     | 2.90e-4 | 4       |
| **ArcFace** | 0.6729     | 4.31e-4 | 3       |
| **SupCon**  | **0.6984** | 3.71e-4 | 6       |

⭐ **SupCon achieved highest validation F1 during HP search**

---

### ✅ Phase 2: 3-Fold Cross-Validation (COMPLETE - 9/9 runs finished)

**Fold 1/3** ✅ **COMPLETE (Apr 9, 12:05 - 14:57 UTC)**
```
Focal Loss (20 epochs):    F1 = 0.7084 ✓ (saved: best_model_focal_fold0.pth)
ArcFace Loss (20 epochs):  F1 = 0.6851 ✓
SupCon Loss (30 epochs):   F1 = 0.7157 ✓
```

**Fold 2/3** ✅ **COMPLETE (Apr 9, 14:57 - 17:30 UTC)**
```
Focal Loss (20 epochs):    F1 = 0.7213 ✓ (saved: best_model_focal_fold1.pth)
ArcFace Loss (20 epochs):  F1 = 0.7300 ✓
SupCon Loss (30 epochs):   F1 = 0.7031 ✓
```

**Fold 3/3** ✅ **COMPLETE (Apr 9, 17:30 - 18:01 UTC)**
```
Focal Loss (20 epochs):    F1 = 0.7191 ✓ (saved: best_model_focal_fold2.pth)
ArcFace Loss (20 epochs):  F1 = 0.7256 ✓
SupCon Loss (30 epochs):   F1 = 0.7432 ✓ (BEST FOLD)
```

---

## 📈 Completion Summary

| Phase | Duration | Status | Completed |
|-------|----------|--------|-----------|
| Phase 1 (HP Search) | ~20 hours | ✅ **COMPLETE** | Apr 9, 11:10 UTC |
| Phase 2 (3-Fold CV) | ~6.5 hours | ✅ **COMPLETE** | Apr 9, 18:01 UTC |
| Phase 3 (Reporting) | ~2 min | ✅ **COMPLETE** | Apr 9, 20:34 UTC |
| **TOTAL** | ~29 hours 10 min | ✅ **COMPLETE** | Apr 9, 20:34 UTC |

---

## 📁 Output Files Generated

**Directory**: `optionB_v3_hpopt_results/` (214 MB total)

```
✅ hp_search_results.txt              1.1 KB   (Best HP per variant)
✅ best_model_focal_fold0.pth        72 MB    (Fold 0 Focal model)
✅ best_model_focal_fold1.pth        72 MB    (Fold 1 Focal model)
✅ best_model_focal_fold2.pth        72 MB    (Fold 2 Focal model)
✅ cv_results.txt                    585 B    (Final CV metrics)
✅ final_comparison.txt              1.7 KB   (HP-tuned vs baseline V3)
```

**Note**: ArcFace and SupCon fold models not saved (Focal used as primary baseline)

---

## 🎯 Final Results & Key Findings

### Phase 1 Optimization Results:
- ✅ SupCon Loss achieved best validation F1: **0.6984** ⭐
- Focal Loss: 0.6833
- ArcFace Loss: 0.6729

### Phase 2 Cross-Validation Results (3-Fold Average):
```
Loss Type    Mean F1   Std Dev    Best Fold    Notes
────────────────────────────────────────────────────
Focal        0.7163    ±0.0056    0.7213      Most stable
ArcFace      0.7136    ±0.0202    0.7300      High variance
SupCon       0.7207    ±0.0167    0.7432      ⭐ BEST OVERALL
```

### Phase 3 Comparison: HP-Tuned vs Baseline V3
```
Loss Type    Baseline   HP-Tuned   Change
────────────────────────────────────────
Focal        0.7736     0.7163     -7.41%
ArcFace      0.7345     0.7136     -2.85%
SupCon       0.7951     0.7207     -9.36%
```

**Key Observation**: Hyperparameter tuning did NOT improve performance vs fixed baseline HP.
The baseline hyperparameters were already well-optimized for this task.
SupCon loss remains the best performer (0.7207 avg) but baseline was superior (0.7951).

---

## ⏱️ To Monitor Progress

```bash
# Check if still running
ps aux | grep optionB_v3_hpopt | grep -v grep | wc -l

# Watch output directory grow
watch -n 5 'ls -lh /home/test/big_data_assignment_2/optionB_v3_hpopt_results/'

# View results as they appear
cat /home/test/big_data_assignment_2/optionB_v3_hpopt_results/hp_search_results.txt
```

---

## 🔧 Implementation Details

**Algorithm**: Bayesian Optimization (Optuna TPE Sampler)  
**Pruning**: Median-based early stopping  
**Parallelization**: 2-3 worker processes  
**Architecture**: EfficientNet-B4 (1792 frozen features)  
**Augmentation**: RandAugment, CutMix, MixUp, RandomErasing, etc.

---

**Last Updated**: Apr 9, 16:09 UTC  
**Next Update**: After Fold 2/3 completes (~17:30 UTC)
