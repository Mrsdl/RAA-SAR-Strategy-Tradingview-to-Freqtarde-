# RAA-SAR TradingView策略实现

## 概述

本策略是基于TradingView Pine Script的RAA-SAR策略的Freqtrade(Windows)实现。该策略结合了RAA（Relative Average Absolute）指标和SAR（Parabolic SAR）指标来进行交易决策。

## 策略逻辑

### 核心思想
- **RAA指标**：用于识别入场时机和波动确认
- **双SAR指标**：用于趋势跟踪和止损保护
- **时间控制**：通过时间间隔控制交易频率

### 入场条件
1. RAA上穿信号点（连续3个方向信号为1）
2. RAA值下降确认（当前RAA值小于前一个或前两个RAA值）

### 出场条件
1. SAR反转信号
2. 长线SAR反转信号
3. 时间间隔控制（gap_time小时后）
4. 1.5倍时间间隔的SAR反转退出
5. 止盈止损保护（5%止盈，2.5%止损）

## 技术指标

### RAA指标
- **FRAMA**：分形自适应移动平均
- **RAA值**：平均绝对偏差
- **上下轨**：FRAMA ± (RAA值 × 倍数)
- **趋势方向**：基于价格与上下轨的关系

### SAR指标
- **普通SAR**：用于主要趋势判断
- **长线SAR**：用于长线趋势确认
- **参数**：起始值、增量、最大值

## 文件结构

```
strategies/
├── user_data/strategies/
│   └── RAA_SAR_Strategy.py  # 主策略文件
├── config_raa_sar.json      # 配置文件
└── README_RAA_SAR_Strategy.md # 说明文档
```

## 使用方法

### 1. 默认已安装freqtrade以及相关依赖
### 2.交易所历史数据下载
```bash
freqtrade download-data -c user_data/config_raa_sar.json --exchange binance --timeframe 1h --timerange 20240901-20250901
```
### 3.回测
```bash
freqtrade backtesting --config user_data/config_raa_sar.json --strategy RAA_SAR_TradingView_Strategy --timeframe 1h --timerange 20240901-20250901
```
### 4.参数优化
```bash
freqtrade hyperopt --config user_data/config_raa_sar.json --strategy RAA_SAR_TradingView_Strategy --hyperopt-Loss SharpeHyperOptLoss --spaces buy roi stoploss --job-workers 1 --epochs 10
```
### 5.虚拟/实盘交易(默认为虚拟交易)
```bash
freqtrade trade --config config_raa_sar_tradingview.json --strategy RAA_SAR_TradingView_Strategy 
```

## 参数设置

### RAA参数
- `raa_length`: RAA计算周期 (默认: 50)
- `raa_mult`: RAA倍数 (默认: 1.35)

### SAR参数
- `sar_af`: SAR起始值 (默认: 0.06)
- `sar_af_increase`: SAR增量 (默认: 0.02)
- `sar_max_af`: SAR最大值 (默认: 0.2)

### 长线SAR参数
- `sar_long_af`: 长线SAR起始值 (默认: 0.001)
- `sar_long_af_increase`: 长线SAR增量 (默认: 0.001)
- `sar_long_max_af`: 长线SAR最大值 (默认: 0.2)

### 时间控制参数
- `gap_time`: 时间间隔（小时）(默认: 12)

## 风险控制

### 止损设置
- **固定止损**: 2.5%
- **动态止损**: 基于SAR趋势和RAA波动性
- **盈利保护**: 根据盈利情况调整止损

### 止盈设置
- **快速止盈**: 5%
- **时间止盈**: 基于持仓时间的递减止盈
- **信号止盈**: 基于技术指标的提前止盈

## 策略特点

### 优势
1. **趋势跟踪**: 基于SAR的趋势跟踪能力
2. **波动适应**: RAA指标适应市场波动性
3. **时间控制**: 避免过度交易
4. **风险控制**: 多层次的风险管理

### 适用市场
- 趋势性较强的市场
- 波动性适中的市场
- 1小时时间框架

### 注意事项
1. 策略在震荡市场中可能产生较多假信号
2. 需要足够的市场流动性
3. 建议在多个交易对上测试

## 监控指标

### 关键指标
- **胜率**: 盈利交易占比
- **盈亏比**: 平均盈利/平均亏损
- **最大回撤**: 最大资金回撤
- **夏普比率**: 风险调整后收益

### 信号质量
- **RAA信号**: 上穿信号的准确性
- **SAR信号**: 反转信号的及时性
- **时间控制**: 持仓时间的合理性

## 优化建议

### 参数优化
1. 使用Freqtrade的hyperopt功能进行参数优化
2. 在不同市场条件下测试参数稳定性
3. 考虑多时间框架的参数设置

### 策略改进
1. 添加更多过滤条件
2. 结合其他技术指标
3. 优化资金管理

## 免责声明

本策略仅供学习和研究使用，不构成投资建议。实际交易中请谨慎使用，并充分了解相关风险。使用本策略进行实盘交易的所有风险由用户自行承担。

## 更新日志

### v1.0.0 (2025-01-01)
- 初始版本发布
- 实现基本的RAA-SAR策略逻辑
- 支持双SAR指标
- 添加时间间隔控制
- 实现止盈止损机制

