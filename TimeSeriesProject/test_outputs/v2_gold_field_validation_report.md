# v2/Gold 字段与行为测试报告

## 测试范围
- 依据文档 `数据处理说明_中美数据下载与清洗.md` 的 v2 + Gold 字段与行为定义进行验证。
- 使用合成的 US 1h 样本，覆盖正常与异常场景（off-calendar、缺失、空值、负值、异常波动、停牌猜测等）。
- 本次仅测试，不修改生产逻辑。

## 数据设计（虚拟问题数据）
- 包含 off-calendar 时间点（盘前）。
- 包含空价格、非正价格、OHLC 不一致、负成交量。
- 人为跳过一个中间 bucket，验证 missing/vendor_gap/suspected_halt。
- 包含 no-trade bar（OHLC 平坦且 volume=0）。
- 人工注入 daily adj_factor 变化，验证 corp_action 标记。

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
| gold has missing placeholders | PASS | count=6 |
| gold missing_reason vendor_gap present | PASS | values=['vendor_gap'] |
| gold has suspected_halt | PASS | count=1 |
| gold has is_no_trade | PASS | count=1 |
| gold has range anomaly flag | PASS | count=2 |
| gold has jump anomaly flag | PASS | count=1 |
| gold has corp-action flag | PASS | count=7 |
| gold masks are binary | PASS | checked feature_mask/label_mask_1/4/7 |

## 发现的问题
- 本次脚本覆盖范围内，未发现功能性失败项。
- 仍有残余风险：真实供应商数据可能出现更复杂异常（停牌跨日、复权源延迟、节假日临时变更）。

## 建议的改进计划（仅建议，不在本次实现）
- 增加参数化测试（按 US/CN、不同 horizon、不同阈值组合自动跑）。
- 增加回归基线：把关键统计（flag 比例、mask 比例）做快照，CI 中比较漂移。
- 增加 edge-case 用例：跨季度边界、早收盘日、连续停牌多天、adj_factor 缺失与异常值。
- 增加自动数据体检脚本入口，定期对最新季度表执行 quality + schema 双重检查。

## 输出文件
- 日志：`/Users/caoy0e/Stock/AI-little-project/TimeSeriesProject/test_outputs/v2_gold_field_validation.log`
- 报告：`/Users/caoy0e/Stock/AI-little-project/TimeSeriesProject/test_outputs/v2_gold_field_validation_report.md`