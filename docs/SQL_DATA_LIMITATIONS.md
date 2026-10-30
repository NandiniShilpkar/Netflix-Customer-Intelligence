# SQL Business Questions — Data Limitations

## Q29: Engagement Before and After Subscription Price Changes

Original requirement:
Calculate average engagement 30 days before and after the most recent subscription price change and percentage change.

Diagnostic findings:
- Total subscription records: 8,875
- Customers with multiple subscriptions: 875
- Customers with plan changes: 570
- Customers with actual monthly_price changes: 0

Observed plan migration records:
- PLAN02 -> PLAN01: 119
- PLAN03 -> PLAN01: 114
- PLAN03 -> PLAN02: 106
- PLAN01 -> PLAN02: 101
- PLAN01 -> PLAN03: 74
- PLAN02 -> PLAN03: 56

Conclusion:
The available dataset contains plan migrations but no recorded monthly price
changes between consecutive subscription records. Therefore, the original
Q29 before/after price-change engagement analysis cannot be measured.

Q29 remains in the SQL question set. Its zero-row result is a documented
data limitation, not a SQL execution failure.

Recommendation:
Collect historical subscription price-change events with effective dates
to support this analysis in a future dataset version.
