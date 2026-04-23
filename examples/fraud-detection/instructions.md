# Fraud Detection Optimization Instructions

## Task
Optimize `train.py` to maximize AUC-ROC for fraud detection on the IEEE-CIS dataset. You may modify both `build_features` (feature engineering) and `train_and_evaluate` (model config). Keep `run_pipeline`'s interface and the `auc_roc: 0.xxxxxx` print format unchanged so the evaluator can parse the metric.

## Dataset Details
- 100K train / 25K val, 3.5% fraud rate, time-based split
- Base data has 297 columns after V-feature correlation pruning
- Categoricals are already label-encoded as integers
- TransactionDT is in seconds (timedelta from reference date, NOT a timestamp)

## Column Meanings (from Kaggle community reverse-engineering)

### Raw columns
- **TransactionAmt**: USD amount. Heavy-tailed (median $68, max $4578). Log transform essential.
- **ProductCD**: Product type (5 categories: C, H, R, S, W). Each has a distinct V-feature NaN pattern and fraud rate (C=11%, W=2.1%).
- **card1**: Bank Identification Number (BIN) — first 6 digits of card. Top-3 importance.
- **card2**: Additional card info. 1.5% NaN. Top-3 importance.
- **card3/card5**: Card country/product type codes.
- **card4**: Card network (visa, mastercard, etc).
- **card6**: Card type (credit, debit).
- **addr1**: Billing zip code (anonymized). 11.5% NaN.
- **addr2**: Billing country.
- **P_emaildomain**: Purchaser email domain (gmail.com, yahoo.com, etc).
- **R_emaildomain**: Recipient email domain. Mismatch between P and R = fraud signal.
- **dist1/dist2**: Distance features.

### C-features (C1-C14): Entity occurrence COUNTS, no NaN
- **C1** (importance rank #2): Count of addresses associated with the payment card
- **C2**: Count of cards at the billing address
- **C5**: Count of email addresses seen with this card
- **C11**: Count of cards associated with a user identity
- **C12**: Count of addresses associated with a user identity
- **C13** (importance rank #4): Count of distinct email domains per entity — **one of the single most predictive raw features**. High values = fraud ring.
- **C14** (importance rank #3): Related count feature

### D-features (D1-D15): TIMEDELTA in days between events
- **D1** (0.2% NaN, median 1 day): Days since last transaction. Most important D-feature. `TransactionDT/86400 - D1` estimates the **account creation date** — this is the key insight for UID construction.
- **D2** (49% NaN, median 97 days): Days since card was first associated with the identity
- **D3** (46% NaN): Days since last similar transaction
- **D4** (29.5% NaN): Days since email association
- **D10** (14% NaN): Days since last device-linked transaction
- **D11** (52% NaN): Days since account was opened / account age
- **D15** (16.5% NaN, median 46 days): Days since last transaction (alternative)
- D-feature NaN rates themselves are informative — missingness patterns encode transaction type

### M-features (M1-M9): Binary MATCH indicators
Whether certain attributes match each other (name↔address, card↔billing, device↔historical, etc). Sum of True values, count of NaN, and the M-vector signature are all useful.

### V-features (V1-V339, ~202 after pruning): Vesta-engineered risk signals
Grouped by ProductCD — each product type uses a different subset of V-features (others are NaN). V258 is the #1 most important feature overall (gain=16703). Other important V-features: V283, V69, V130, V307, V294, V201.

## Top Winning Techniques (from 1st-3rd place solutions)

### 1. UID Construction (THE most impactful single technique)
```python
D1_start = floor(TransactionDT / 86400 - D1)  # estimated account creation day
uid = card1 + "_" + addr1 + "_" + D1_start
```
This creates a stable user fingerprint. All aggregation features should be computed on this UID.

### 2. UID-level aggregation features
For each UID, compute: mean, std, count of TransactionAmt. Then z-score and ratio for each transaction relative to user's history. This captures "is this transaction unusual for this user?"

### 3. Temporal centroid distance
Compute the user's typical time-of-day using cyclical hour_sin/hour_cos means. The Euclidean distance of the current transaction from the centroid = "is this at an unusual time for this user?"

### 4. D-feature lifecycle lags
D1 - D2, D1 - D4, D1 - D10, D1 - D15: Inconsistencies between these timestamps indicate synthetic identities or account takeovers.

### 5. Velocity features (sort by [uid, TransactionDT])
Time since last transaction per user. Amount change from previous transaction. High velocity + high amount = fraud signal.

### 6. Cross-entity cardinality (nunique)
How many unique addr1 values per card1? How many unique card1 per addr1? How many unique P_emaildomain per uid? High cardinality = suspicious.

### 7. NaN pattern signature
The binary NaN/not-NaN pattern across D+M columns encodes the transaction type. Compute a bitwise signature or just count NaN per feature group.

### 8. Frequency encoding
For card1, card2, addr1, P_emaildomain, etc. — map each value to its frequency. Rare values (appearing once or twice) are fraud signals.

### 9. Interaction features
- amount_zscore × time_distance (unusual amount at unusual time)
- amount_zscore × C1_ratio (unusual amount with unusual address count)
- amount / (D1 + 1) = spending rate per day since last transaction

### 10. Row-wise missingness features
Count of NaN values across D-columns, M-columns, V-columns per row. Sum/mean of M-column values. The NaN pattern encodes the transaction profile.

## Important Constraints
- Keep code under 300 lines (Weco backend limit)
- Use n_jobs=4 for any model operations
- `train.py` loads `data/base_train_small.parquet` and `data/base_val_small.parquet` — don't change these paths
- Categoricals are already integer-encoded — treat them as numeric
- Keep the `run_pipeline() -> float` function signature and the `auc_roc: 0.xxxxxx` print format intact

## Avoiding silent target leakage
`isFraud` is the label. If you compute features that aggregate across all columns of the dataframe (e.g. `(df == 0).sum(axis=1)`, row-wise NaN counts over the entire frame), drop `isFraud` and `TransactionID` first. Otherwise the label signal bleeds into the features and produces implausibly high AUC (>0.95) that collapses the moment the fix is applied. Target encoding must use out-of-fold protection: compute encoding on train folds only, never on the full train + val concat.
