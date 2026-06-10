# Feature Registry

| Name | Group | Type | Source Columns | Leakage Risk | Default | Owner | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amount_log1p | transaction | double | amount | low | 0 | risk-ml | Natural log of amount plus one. |
| origin_balance_delta | transaction | double | oldbalanceOrg, newbalanceOrig | low | 0 | risk-ml | Origin account balance delta across the transaction. |
| destination_balance_delta | transaction | double | oldbalanceDest, newbalanceDest | low | 0 | risk-ml | Destination account balance delta across the transaction. |
| origin_delta_minus_amount | transaction | double | oldbalanceOrg, newbalanceOrig, amount | low | 0 | risk-ml | Origin balance delta relative to stated amount. |
| destination_delta_minus_amount | transaction | double | oldbalanceDest, newbalanceDest, amount | low | 0 | risk-ml | Destination balance delta relative to stated amount. |
| origin_balance_was_zero | transaction | int | oldbalanceOrg | low | 0 | risk-ml | Origin balance was zero before the transaction. |
| origin_balance_is_zero | transaction | int | newbalanceOrig | low | 0 | risk-ml | Origin balance is zero after the transaction. |
| destination_balance_was_zero | transaction | int | oldbalanceDest | low | 0 | risk-ml | Destination balance was zero before the transaction. |
| destination_balance_is_zero | transaction | int | newbalanceDest | low | 0 | risk-ml | Destination balance is zero after the transaction. |
| is_cash_out | transaction | int | type | low | 0 | risk-ml | Transaction type is CASH_OUT. |
| is_transfer | transaction | int | type | low | 0 | risk-ml | Transaction type is TRANSFER. |
| is_payment | transaction | int | type | low | 0 | risk-ml | Transaction type is PAYMENT. |
| is_debit | transaction | int | type | low | 0 | risk-ml | Transaction type is DEBIT. |
| is_cash_in | transaction | int | type | low | 0 | risk-ml | Transaction type is CASH_IN. |
| is_merchant_destination | transaction | int | nameDest | low | 0 | risk-ml | Destination account name appears to be a merchant. |
| large_amount_flag | transaction | int | amount | low | 0 | risk-ml | Amount exceeds the deterministic large-amount threshold. |
| previous_step_by_origin | temporal | bigint | nameOrig, step | low |  | risk-ml | Previous observed transaction step for the origin entity. |
| steps_since_previous_origin_tx | temporal | bigint | nameOrig, step | low | -1 | risk-ml | Elapsed steps since the previous origin transaction. |
| origin_tx_count_before | temporal | bigint | nameOrig, step | low | 0 | risk-ml | Count of earlier transactions by the origin entity. |
| origin_amount_mean_before | temporal | double | nameOrig, step, amount | low | 0.0 | risk-ml | Mean prior transaction amount for the origin entity. |
| origin_amount_std_before | temporal | double | nameOrig, step, amount | low | 0.0 | risk-ml | Population standard deviation of prior origin amounts. |
| origin_amount_to_mean_ratio | temporal | double | nameOrig, step, amount | low | 0.0 | risk-ml | Current amount divided by the prior origin mean amount. |
| origin_amount_zscore_before | temporal | double | nameOrig, step, amount | low | 0.0 | risk-ml | Current amount z-score against prior origin history. |
| origin_total_tx_count | entity | bigint | nameOrig | low | 0 | risk-ml | Total transactions observed for the origin entity. |
| origin_avg_amount | entity | double | nameOrig, amount | low | 0.0 | risk-ml | Average amount sent by the origin entity. |
| origin_max_amount | entity | double | nameOrig, amount | low | 0.0 | risk-ml | Maximum amount sent by the origin entity. |
| origin_unique_destinations | entity | bigint | nameOrig, nameDest | low | 0 | risk-ml | Distinct destinations used by the origin entity. |
| destination_total_tx_count | entity | bigint | nameDest | low | 0 | risk-ml | Total transactions received by the destination entity. |
| destination_avg_amount | entity | double | nameDest, amount | low | 0.0 | risk-ml | Average amount received by the destination entity. |
| destination_max_amount | entity | double | nameDest, amount | low | 0.0 | risk-ml | Maximum amount received by the destination entity. |
| destination_unique_origins | entity | bigint | nameOrig, nameDest | low | 0 | risk-ml | Distinct originators that have sent to the destination entity. |
| origin_destination_pair_count | entity | bigint | nameOrig, nameDest | low | 0 | risk-ml | Observed transaction count for the origin-destination pair. |
| origin_destination_avg_amount | entity | double | nameOrig, nameDest, amount | low | 0.0 | risk-ml | Average amount for the origin-destination pair. |
| origin_out_degree | graph | bigint | nameOrig, nameDest | low | 0 | risk-ml | Count of distinct destination nodes connected to the origin node. |
| destination_in_degree | graph | bigint | nameOrig, nameDest | low | 0 | risk-ml | Count of distinct origin nodes connected to the destination node. |
| edge_frequency | graph | bigint | nameOrig, nameDest | low | 0 | risk-ml | Observed frequency of the origin-destination edge. |
| destination_historical_fraud_rate | graph | double | nameDest, isFraud | medium | 0.0 | risk-ml | Historical label rate for the destination node. |
| destination_historical_fraud_count | graph | bigint | nameDest, isFraud | medium | 0 | risk-ml | Historical fraud count for the destination node. |
| has_card1 | identity | int | card1 | low | 0 | risk-ml | card1 field is present. |
| has_card2 | identity | int | card2 | low | 0 | risk-ml | card2 field is present. |
| has_card3 | identity | int | card3 | low | 0 | risk-ml | card3 field is present. |
| has_card4 | identity | int | card4 | low | 0 | risk-ml | card4 field is present. |
| has_card5 | identity | int | card5 | low | 0 | risk-ml | card5 field is present. |
| has_card6 | identity | int | card6 | low | 0 | risk-ml | card6 field is present. |
| p_emaildomain_normalized | identity | string | P_emaildomain | low | unknown | risk-ml | Normalized purchaser email domain. |
| r_emaildomain_normalized | identity | string | R_emaildomain | low | unknown | risk-ml | Normalized recipient email domain. |
| has_p_emaildomain | identity | int | P_emaildomain | low | 0 | risk-ml | Purchaser email domain is present. |
| has_r_emaildomain | identity | int | R_emaildomain | low | 0 | risk-ml | Recipient email domain is present. |
| email_domains_match | identity | int | P_emaildomain, R_emaildomain | low | 0 | risk-ml | Purchaser and recipient email domains match after normalization. |
| device_type_normalized | identity | string | DeviceType | low | unknown | risk-ml | Normalized IEEE-CIS device type. |
| device_info_normalized | identity | string | DeviceInfo | low | unknown | risk-ml | Normalized IEEE-CIS device info string. |
| identity_missing_value_count | identity | int | id_*, card*, addr*, dist*, M*, P_emaildomain, R_emaildomain, DeviceType, DeviceInfo | low | 0 | risk-ml | Count of missing values across present identity-related columns. |
| product_cd_avg_transaction_amount | identity | double | ProductCD, TransactionAmt | low | 0.0 | risk-ml | Average transaction amount for the product code. |
| product_cd_max_transaction_amount | identity | double | ProductCD, TransactionAmt | low | 0.0 | risk-ml | Maximum transaction amount for the product code. |
| product_cd_transaction_count | identity | bigint | ProductCD | low | 0 | risk-ml | Transaction count observed for the product code. |
