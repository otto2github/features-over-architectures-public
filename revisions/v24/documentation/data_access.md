# Data access and source identity

## Availability levels
The supplementary archive supplies source code, field inventories, hashes, fitted-run metrics and aggregate diagnostics. It does not distribute licensed source rows, company-level scores/masks or person identifiers. Third-party access is obtained from the providers, not presumed from this ZIP. The absence of an independent DOI is not presented as an approved third-party-data exception.

## CSMAR
Provider: China Stock Market & Accounting Research, https://www.csmar.com ; platform https://data.csmar.com . Use an individual or institutional license that authorizes the relevant research access. The retained project export identifiers are STK_Violation_Main, FIN_Audit, PLED_TRDDETL, HLD_Contrshr and RPT_Operation (three retained volumes). They cover violation/sanction records, audit reports, pledge transactions, controllers and related-party operations. Table identifiers are the aliases in the supplied source, not a guarantee that every current commercial product uses the same display name.

The authors' research-use authorization is stated in the manuscript. The contract version, redistribution rights for scores/derivatives, and original query/extraction timestamps are not supplied as independently verified metadata. No private derivative redistribution right is inferred. An independent researcher should confirm table access and permitted use with their institution/provider.

## Tushare
Provider/API documentation: https://tushare.pro/document/1?doc_id=13 . Endpoints: income, balancesheet, cashflow, fina_indicator, stock_basic, top10_holders, top10_floatholders and stk_managers. Users need their own account/token and the corresponding endpoint permissions. Financial schema examples: https://tushare.pro/document/2?doc_id=33 (income) and https://tushare.pro/document/2?doc_id=79 (financial indicators). No credential is distributed.

`raw_input_fields_v20.csv` lists recorded source fields; `feature_input_inventory_v8.csv` and the source inventory map model columns to source evidence. Date/version columns are distinguished from numerical values. In particular comp_type is not by itself a consolidated-report flag. A current API response cannot be assumed to restore every historically available vendor version.

## CNINFO
Public portal: https://www.cninfo.com.cn . The label-construction programs retain the actual annual-report announcement queries, title/date matching and fallback rules. Relevant fields include secCode, announcementTitle, announcementTime, announcementId and adjunctUrl. Independent use remains subject to the portal's current access conditions.

## Analysis dates versus acquisition dates
Panel fiscal years are 2010–2024; primary eligible train/validation/test years are 2010–2018, 2019–2020 and 2021–2022. The common enforcement cutoff is 8 May 2026. The source-record checks executed on 24 September 2026 used financials.db SHA-256 `5cc0550b95c7bbd44fb9d819096ec4b4cb5827839a837af2210d7fc69a3d30cb`; frozen node_features_v1_1.parquet is `adc00129f06ffb0995691c6d1ba98f740c53561c1e0cc4e91d927b84eb36beb2`; the label hash is `1d50378138dfc65def76555c0a48aa9cea2656af4d456fe5178d9cf7eae31836`.

These are file/analysis identities, not original acquisition timestamps. Exact original API query dates, source contract identifiers, and complete historical vintages remain unestablished. File modification times are not substituted for them. Re-querying the current vendors may reproduce the processing schema, not these exact frozen bytes. The source code and fingerprint lists identify which additional authorized inputs are required.

## Two different financial compatibility measures
`aggregates/subset_diagnostics/financial_masks/13_financial_value_binding.csv` reports partition/label-stratified value binding from the source audit, including requests for nine upstream raw per-share intermediates not retained in the model schema. `aggregates/preparation/asof_current_snapshot_replay_compatibility.csv` measures the separate retained-feature replay with its finite-value denominators and normalized-field diagnostics. Their denominators, selected fields and scope differ. A 0.75 subgroup rate in the former is not a contradiction of the 0.9977/0.9933 raw-feature minima in the latter. Neither establishes historical completeness.

Official access descriptions were checked on 25 September 2026; they do not replace the study's actual licensing agreement or unpublished extraction records.
