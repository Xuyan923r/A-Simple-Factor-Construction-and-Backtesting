# 量化因子回测系统 (Quantitative Factor Backtesting System)

## 项目概述

这是一个完整的量化因子回测系统，用于计算、测试和评估各种量化因子的有效性。系统支持多种因子类型，包括量价因子、技术指标因子等，并提供了完整的回测框架和结果分析功能。

## 项目结构

```
quant_mm/
├── factor/                          # 因子相关文件
│   ├── alpha/                       # 因子数据文件
│   │   ├── alpha008.parquet
│   │   ├── alpha019.parquet
│   │   ├── alpha022.parquet
│   │   ├── alpha032.parquet
│   │   ├── alpha053.parquet
│   │   └── alpha054.parquet
│   ├── Factor_construct/            # 因子构建示例
│   │   ├── alpha008.ipynb
│   │   ├── alpha019.ipynb
│   │   ├── alpha022.ipynb
│   │   ├── alpha032.ipynb
│   │   ├── alpha053.ipynb
│   │   └── alpha054.ipynb
│   ├── data/                        # 原始数据
│   │   ├── merge_daily_info.parquet
│   │   ├── TRD_Dalyr.parquet
│   │   └── data_statement.xlsx
│   ├── alpha101.pdf                 # 因子说明文档
│   ├── alpha101.xlsx                # 因子数据表
│   ├── calc_cvturn.parquet          # 计算中间数据
│   └── 数据指标.txt                 # 数据字段说明
├── result/                          # 回测结果
│   ├── alpha008/                    # 各因子回测结果
│   ├── alpha019/
│   ├── alpha022/
│   ├── alpha032/
│   ├── alpha053/
│   └── alpha054/
├── 单因子回测函数1213.py            # 主要回测函数
├── 单因子回测函数1213 copy.py       # 回测函数副本
├── hc.py                           # 优化版回测函数
├── calculate.py                    # 指标计算工具
├── Debug.py                        # 调试工具
├── adj_close.parquet               # 复权收盘价数据
├── mv_turn.parquet                 # 市值换手率数据
├── Suspension_Limit.parquet        # 停牌涨跌停数据
└── history-1800.pq                 # 股票池数据
```

## 核心功能

### 1. 因子构建 (Factor Construction)
- 支持基于量价数据的因子计算
- 提供标准化的因子构建流程
- 支持多种技术指标和统计方法

### 2. 因子处理 (Factor Processing)
- **标准化处理**: 使用Z-score方法对因子进行标准化
- **市值中性化**: 通过回归残差方法去除市值影响
- **数据清洗**: 处理缺失值、异常值和无穷值
- **股票池筛选**: 支持自定义股票池，自动排除创业板和科创板

### 3. 回测框架 (Backtesting Framework)
- **多频率回测**: 支持日频(D)、周频(W)、月频(M)调仓
- **分组测试**: 10分组回测，计算各分组收益率
- **交易成本**: 考虑交易费用、印花税等成本
- **停牌处理**: 自动处理停牌和涨跌停股票

### 4. 结果分析 (Result Analysis)
- **收益率指标**: 累计收益率、年化收益率
- **风险指标**: 年化标准差、最大回撤
- **绩效指标**: 夏普比率、单调性分析
- **可视化**: 净值曲线图、分组对比图

## 数据说明

### 主要数据文件
- `adj_close.parquet`: 复权收盘价数据
- `mv_turn.parquet`: 市值和换手率数据
- `Suspension_Limit.parquet`: 停牌和涨跌停状态数据
- `history-1800.pq`: 股票池数据（1800只股票）

### 数据字段说明
- `Stkcd`: 证券代码
- `TradingDate`: 交易日期
- `Opnprc`: 开盘价
- `Hiprc`: 最高价
- `Loprc`: 最低价
- `Clsprc`: 收盘价
- `Dnshrtrd`: 成交股数
- `Dnvaltrd`: 成交金额
- `Dsmvosd`: 流通市值
- `Dsmvtll`: 总市值
- `Dretwd`: 考虑分红的收益率
- `Dretnd`: 不考虑分红的收益率

## 使用方法

### 1. 环境准备
```bash
# 安装必要的Python包
pip install pandas numpy scipy matplotlib statsmodels pyarrow tqdm
```

### 2. 运行回测
```python
# 使用主要回测函数
python 单因子回测函数1213.py

# 或使用优化版回测函数
python hc.py
```

### 3. 自定义因子回测
```python
from 单因子回测函数1213 import back_test

# 创建回测实例
test = back_test(
    factor=factor_data,           # 因子数据
    price=price_data,             # 价格数据
    fac_name='factor_name',       # 因子名称
    dir='result_path',            # 结果保存路径
    market_index='1800',          # 市场指数
    Suspension_Limit=suspension_data,  # 停牌数据
    start_date='2010-01-01',      # 开始日期
    end_date='2025-06-30'         # 结束日期
)

# 因子处理
test.factor_process(
    stock_pool=stock_pool,        # 股票池
    market_value=market_value,    # 市值数据
    neutralize='cap'              # 中性化方式
)

# 分组回测
test.group_test(freq='W', save=True, period='all')

# 结果分析
test.aly_group_test(freq='W', save=True, period='all')
```

## 回测结果

### 输出文件
每个因子会生成以下结果文件：
- `分组净值.csv`: 各分组净值数据
- `分组收益率.csv`: 各分组收益率数据
- `分组回测结果.csv`: 综合回测指标
- `分组净值.png`: 净值曲线图

### 关键指标
- **累计收益率 (CumulativeReturn)**: 整个回测期的总收益率
- **年化收益率 (anlYield)**: 年化后的收益率
- **年化标准差 (anlStd)**: 年化后的波动率
- **夏普比率 (Sharp)**: 风险调整后收益
- **最大回撤 (MaxDrawdown)**: 最大回撤幅度
- **单调性**: 因子值与收益的单调关系

## 因子示例

### Alpha008因子
```python
# 因子计算公式
# 1. 计算日收益率
daily_returns = price.pct_change()

# 2. 计算过去5日开盘价之和
sum_open_5 = open_price.rolling(5).sum()

# 3. 计算过去5日收益率之和
sum_returns_5 = daily_returns.rolling(5).sum()

# 4. 计算乘积项
product_term = sum_open_5 * sum_returns_5

# 5. 计算10日延迟差值
inner_diff = product_term - product_term.shift(10)

# 6. 排名和标准化
ranked_diff = inner_diff.rank()
factor = -1 * (ranked_diff - ranked_diff.mean()) / ranked_diff.std()
```

## 注意事项

1. **数据路径**: 请根据实际情况修改数据文件路径
2. **股票池**: 系统默认使用1800只主板股票，自动排除创业板和科创板
3. **交易成本**: 默认考虑0.1%的交易费用和0.01%的印花税
4. **调仓频率**: 建议使用周频调仓，避免过度交易
5. **中性化**: 可选择是否进行市值中性化处理

## 技术特点

- **高效处理**: 使用pandas和numpy进行向量化计算
- **内存优化**: 采用分块处理，避免内存溢出
- **进度显示**: 使用tqdm显示处理进度
- **错误处理**: 完善的异常处理机制
- **结果可视化**: 自动生成图表和报告

## 扩展功能

- 支持多因子组合回测
- 支持不同市场环境下的表现分析
- 支持因子有效性检验
- 支持自定义交易策略

## 联系方式

如有问题或建议，请联系项目维护者。 