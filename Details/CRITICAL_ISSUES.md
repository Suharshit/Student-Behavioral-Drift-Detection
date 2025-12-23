# Critical Issues & Code Discrepancies

## ⚠️ CRITICAL ISSUES FOUND IN CODE

### Issue 1: Feature Mismatch in Clustering
**Location**: `app.py` line 53

**Problem**:
```python
cluster = kmeans.predict(features_scaled[:, :5])[0]
```
- Uses only first 5 features: `[studytime, absences, failures, G1, G2]`
- But K-Means was trained on: `[studytime, absences, failures, grade_change_1_2, grade_change_2_3]`

**Impact**: Cluster assignments may be incorrect!

**Fix Required**:
```python
# Calculate cluster features properly
cluster_features = np.array([[studytime, absences, failures, 
                              grade_change_1_2, grade_change_2_3]])
cluster_features_scaled = scaler_cluster.transform(cluster_features)
cluster = kmeans.predict(cluster_features_scaled)[0]
```

---

### Issue 2: Missing Cluster Values in Feature Array
**Location**: `app.py` lines 42, 46

**Problem**:
```python
behavior_cluster = 0     # Placeholder (not calculated)
cluster_drift = 0        # Placeholder (not calculated)
```
- Models were trained with actual `behavior_cluster` and `cluster_drift` values
- App sets them to 0 (default)

**Impact**: Predictions may be inaccurate because models expect real cluster values!

**Fix Required**:
1. Calculate `behavior_cluster` using K-Means (see Issue 1)
2. Calculate `cluster_drift` by comparing early and mid clusters
3. Use actual values in feature array

---

### Issue 3: grade_change_2_3 Set to Zero
**Location**: `app.py` line 41

**Problem**:
```python
grade_change_2_3 = 0  # unknown, predicted indirectly
```
- Model expects actual `grade_change_2_3` value
- Set to 0 because G3 hasn't occurred yet

**Impact**: 
- This is actually **correct** for prediction scenario (G3 is unknown)
- But model was trained with actual values
- May reduce prediction accuracy

**Note**: This is a design limitation, not a bug. Consider training models without `grade_change_2_3` for true prediction.

---

### Issue 4: No Clipping for Predicted Grades
**Location**: `app.py` line 69

**Problem**:
```python
final_grade=round(final_grade, 2)
```
- No validation that `final_grade` is between 0 and 20
- Linear regression can predict values outside valid range

**Impact**: May display unrealistic grades (e.g., -2.5 or 25.3)

**Fix Required**:
```python
final_grade = max(0, min(20, regressor.predict(features_scaled)[0]))
final_grade = round(final_grade, 2)
```

---

### Issue 5: Perfect Model Scores Suggest Overfitting
**Location**: `CA2 Predictive.ipynb`

**Problem**:
- Logistic Regression: 100% accuracy
- Linear Regression: R² = 1.0, RMSE ≈ 0

**Impact**: 
- Models likely overfitted to training data
- May not generalize to new data
- Suggests possible data leakage

**Explanation Needed**:
- Strong correlation between G1, G2, and G3
- Small dataset (395 samples)
- Feature engineering captures most variance
- Need cross-validation to verify

---

### Issue 6: No Model Versioning or Update Mechanism
**Location**: Entire application

**Problem**:
- Models are static (loaded once at startup)
- No mechanism to retrain with new data
- No versioning system

**Impact**: 
- Model performance may degrade over time
- Cannot adapt to changing student patterns

**Future Improvement**: Implement model retraining pipeline

---

## 🔧 RECOMMENDED FIXES

### Priority 1 (Critical - Fix Before Presentation)
1. ✅ Fix clustering feature mismatch
2. ✅ Calculate actual cluster values
3. ✅ Add grade clipping (0-20 range)

### Priority 2 (Important - Discuss in Presentation)
4. ⚠️ Acknowledge perfect scores limitation
5. ⚠️ Explain grade_change_2_3 design choice
6. ⚠️ Discuss overfitting concerns

### Priority 3 (Future Improvements)
7. 🔄 Implement cross-validation
8. 🔄 Add model versioning
9. 🔄 Create retraining pipeline

---

## 📋 HOW TO EXPLAIN THESE ISSUES IN VIVA

### If Asked About Perfect Scores:
**Answer**: "The perfect scores (100% accuracy, R²=1.0) are likely due to:
1. Strong correlation between G1, G2, and G3
2. Small dataset size (395 samples)
3. Feature engineering capturing most variance
4. Possible overfitting

To address this, I would:
- Implement cross-validation
- Use regularization techniques
- Collect more data
- Remove highly correlated features"

### If Asked About Feature Mismatch:
**Answer**: "I've identified a discrepancy where the clustering uses different features in the app versus training. This is a known limitation that should be fixed by:
1. Using the same features for clustering as in training
2. Properly calculating cluster values before prediction
3. Ensuring feature consistency across the pipeline"

### If Asked About grade_change_2_3:
**Answer**: "Since G3 hasn't occurred yet during prediction, grade_change_2_3 is set to 0. This is a design choice, but it means the model is trained with a feature it won't have during real prediction. A better approach would be to train models without grade_change_2_3 for true prediction scenarios."

---

## ✅ PRE-PRESENTATION CHECKLIST

- [ ] Understand all issues listed above
- [ ] Prepare explanations for each issue
- [ ] Know how to fix each issue (even if not implemented)
- [ ] Be ready to discuss limitations honestly
- [ ] Have improvement suggestions ready
- [ ] Test application with various inputs
- [ ] Document edge cases encountered

