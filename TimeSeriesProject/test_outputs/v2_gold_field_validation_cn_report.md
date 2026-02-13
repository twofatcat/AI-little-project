# CN v2/Gold 字段与行为测试报告

## 测试范围
- 验证 CN 市场 v2/Gold 字段与行为。
- 重点覆盖 CN 特性：午休断档、`bar_end` 时间语义、交易日内缺失占位。
- 本次仅测试，不修改生产代码。

## 检查结果
- 总检查项：15
- 通过：15
- 失败：0

| 检查项 | 结果 | 说明 |
|---|---|---|
| v2 required fields present | PASS | missing=[] |
| gold required fields present | PASS | missing=[] |
| v2 has off-calendar flag | PASS | count=1 |
| v2 has null-price flag | PASS | count=1 |
| v2 has nonpositive-price flag | PASS | count=1 |
| v2 has ohlc-inconsistent flag | PASS | count=2 |
| v2 has volume-negative flag | PASS | count=1 |
| gold has missing placeholders | PASS | count=1 |
| gold missing_reason vendor_gap present | PASS | values=['vendor_gap'] |
| gold has suspected_halt | PASS | count=1 |
| gold has is_no_trade | PASS | count=1 |
| gold has range anomaly flag | PASS | count=1 |
| gold has corp-action flag | PASS | count=4 |
| gold masks are binary | PASS | checked feature_mask/label_mask_1/4/7 |
| CN lunch break buckets respected | PASS | expect end times include 10:30/11:30/14:00/15:00 |

## 发现的问题
- 本次覆盖范围内未发现功能性失败。
- 残余风险：真实行情下的长停牌、节假日临时调整、供应商复权口径差异仍需在线回归验证。

## 建议改进（仅建议）
- 增加 CN 专属回归集：春节前后、节前半天、长假后首日等样本。
- 增加 source 对比测试（Sina/Eastmoney）并输出差异统计。
- 在 CI 增加 Gold masks 稳定性监控（各 symbol 的 mask 比例阈值报警）。

## 输出文件
- 日志：`/Users/caoy0e/Stock/AI-little-project/TimeSeriesProject/test_outputs/v2_gold_field_validation_cn.log`
- 报告：`/Users/caoy0e/Stock/AI-little-project/TimeSeriesProject/test_outputs/v2_gold_field_validation_cn_report.md`