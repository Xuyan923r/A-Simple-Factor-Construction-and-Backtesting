import pandas as pd
import os
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from scipy.stats import norm
import scipy.stats as stats
import statsmodels.api as sm
import time
import pyarrow.parquet as pq
from tqdm import tqdm


# 解决中文乱码 (Removed, as all output text will be English)
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["font.family"] = "sans-serif"
# 解决负号无法显示的问题
import matplotlib.pyplot as plt

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'PingFang SC', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解决负号 '-' 显示为方块的问题


import warnings
warnings.filterwarnings('ignore')

def calculate_runtime(func):
    """计算函数运行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"{func.__name__} Runtime: {runtime:.2f} seconds\n") # 输出英文
        return result
    return wrapper


class back_test:
    """
    单因子回测类
    """
    def __init__(self,factor,price,fac_name,dir,market_index,Suspension_Limit,start_date='2010-01-01',end_date='2025-06-30'):
        """
            factor：因子值数据集。
            price：收盘价数据集。
            fac_name：因子名称
            dir：存储结果的目录路径。
            market_index：市场指数名称。
            Suspension_Limit：停牌和涨跌停状态表,Suspension=1表示停牌，LimitStatus=1表示涨停，LimitStatus=-1表示跌停
            start_date 和 end_date：回测的起止日期。
            weight：加权方法，默认使用市值加权。(此参数未直接在init中使用，但保留原注释风格)
        """       
        # 确保 fac_name 存在于 factor 的列中
        if fac_name not in factor.columns:
            raise KeyError(f"Column '{fac_name}' not found in factor data. Please check if the column name in your factor file matches the filename.")

        factor[fac_name] = factor[fac_name].astype(float)
        if 'Accper' in factor.columns:
            factor.rename(columns={'Accper': 'TradingDate'}, inplace=True)
        factor['TradingDate'] = pd.to_datetime(factor['TradingDate'])
        factor=factor[(factor['TradingDate']>=start_date) & (factor['TradingDate']<=end_date)]

        price['TradingDate'] =pd.to_datetime(price['TradingDate'])
        price=price[(price['TradingDate']>=start_date) & (price['TradingDate']<=end_date)]

        Suspension_Limit['TradingDate'] = pd.to_datetime( Suspension_Limit['TradingDate'])
        Suspension_Limit=Suspension_Limit[(Suspension_Limit['TradingDate']>=start_date) & (Suspension_Limit['TradingDate']<=end_date)]
        # 对齐为下一日的交易限制状态
        Suspension_Limit[['Suspension', 'LimitStatus']] = (Suspension_Limit.groupby('Stkcd')[['Suspension', 'LimitStatus']].shift(-1)) 

        self.start_date=start_date # 开始时间
        self.end_date= end_date # 结束时间
        self.price = price      # 存储收盘价   
        self.fac_name = fac_name   # 测试单因子名称
        self.factor=factor         # 因子值
        self.dir=dir               # 存储位置
        self.Suspension_Limit = Suspension_Limit  # 股票交易限制状态（1：受限  0：正常）
        self.neutralize = None     # 因子中性化方式
        self.market_index_name=market_index # 市场指数名称
        if not os.path.exists(dir):
            os.makedirs(dir)

    @ calculate_runtime
    def factor_process(self,stock_pool,market_value=None,neutralize=None):        # 对已获取的因子值进行处理，进行后续操作和计算
        # step 0: 股票池更迭
        stock_pool = stock_pool.loc[self.start_date:self.end_date]
        # 确保stock_pool的列名是整数
        stock_pool.columns = [int(str(col).split('.')[0]) for col in stock_pool.columns] 
        stock_pool.index = pd.to_datetime(stock_pool.index)

        # step 0.5：排除创业板、科创板股票  
        columns_to_zero = [col for col in stock_pool.columns if (300000 <= col <= 301999) or (688000 <= col <= 689999)]  
        stock_pool.loc[:, columns_to_zero] = 0  

        # 只保留在股票池中的数据
        stock_pool = stock_pool.stack().reset_index()
        stock_pool.columns = ['TradingDate', 'Stkcd', 'in_pool']
        
        # 优化合并操作，避免不必要的DataFrame复制
        self.factor = pd.merge(self.factor, stock_pool, on=['TradingDate','Stkcd'], how='inner') # 只保留在股票池内的因子值
        self.factor = self.factor[self.factor['in_pool'] == 1].drop(columns='in_pool')

        self.Suspension_Limit = pd.merge(self.Suspension_Limit, stock_pool, on=['TradingDate','Stkcd'], how='inner')
        self.Suspension_Limit = self.Suspension_Limit[self.Suspension_Limit['in_pool'] == 1].drop(columns='in_pool')
        
        self.price = pd.merge(self.price, stock_pool, on=['TradingDate','Stkcd'], how='inner')
        self.price = self.price[self.price['in_pool'] == 1].drop(columns='in_pool')


        # Step 1: 因子数据进行基本处理，删除 NaN 和 inf 值
        self.factor[self.fac_name] = self.factor[self.fac_name].replace([np.inf, -np.inf], np.nan).dropna()

        # step 2: 在每个截面上对因子标准化   这里用的是 z-score 标准化
        # 避免在 apply 中 reset_index，直接赋值给原列
        self.factor[self.fac_name] = self.factor.groupby('TradingDate')[self.fac_name].transform(lambda x: (x - x.mean()) / x.std())
        # 在标准化后再次处理NaN值，因为某些分组可能标准差为0导致NaN
        self.factor[self.fac_name] = self.factor[self.fac_name].replace([np.inf, -np.inf], np.nan).dropna()


        # step 3：市值中性化处理
        self.neutralize = neutralize
        if neutralize == 'cap':            
            # 确保 market_value 中性化所需列存在且类型正确
            if market_value is None:
                raise ValueError("market_value data must be provided for market capitalization neutralization.")
            
            df_for_neutralize = pd.merge(self.factor, market_value, on=['TradingDate','Stkcd'],how='inner')
            # 确保市值数据没有NaN或0
            df_for_neutralize = df_for_neutralize.dropna(subset=['MarketValue'])
            df_for_neutralize = df_for_neutralize[df_for_neutralize['MarketValue'] > 0]

            if df_for_neutralize.empty:
                print(f"Warning: Merged data is empty for {self.fac_name} during market capitalization neutralization. Skipping neutralization.") # 输出英文
                return 

            # 对每个交易日期和股票进行市值中性化处理
            def neutralize_by_cap(group):
                group['const'] = 1
                group['log_mv'] = np.log(group['MarketValue'])  # 对市值进行对数处理

                # 回归残差的方式进行市值中性化
                X = group[['log_mv', 'const']]
                y = group[self.fac_name]
                
                model = sm.OLS(y, X).fit()
                group['neutralized_factor'] = model.resid  # 得到市值中性化后的因子

                return group[['TradingDate', 'Stkcd', 'neutralized_factor']]

            # 按照交易日期分组并应用市值中性化
            # 按照交易日期分组并应用市值中性化
            df_neutralized = df_for_neutralize.groupby('TradingDate').apply(neutralize_by_cap)
            # 将中性化后的因子赋值回原因子
            self.factor = df_neutralized.rename(columns={'neutralized_factor': self.fac_name})


    @ calculate_runtime
    def group_test(self,freq='W',save=False,period='all'):                 # 分组净值测试 （主要保留周频的）
        """
        freq   调仓周期    日频D 周频W 月频M
        method 分组加权方法
        direction 因子的正负向
        """
        if freq=='D':
            df=self.factor.copy()
            df=df.merge(self.Suspension_Limit,on=['TradingDate','Stkcd'],how='right')
            df=df.merge(self.price, on=['TradingDate', 'Stkcd'], how='right')
        elif freq=='W':  # 每周五进行调仓
            price=self.price.groupby([pd.Grouper(key='TradingDate',freq='W-Fri'),'Stkcd'])['price'].last().reset_index()
            factor= self.factor.groupby([pd.Grouper(key='TradingDate',freq='W-Fri'),'Stkcd'])[self.fac_name].last().reset_index()
            Suspension_Limit=self.Suspension_Limit.groupby([pd.Grouper(key='TradingDate', freq='W-Fri'), 'Stkcd'])[['Suspension', 'LimitStatus']].last().reset_index()  
            df=factor.merge(Suspension_Limit,on=['TradingDate','Stkcd'],how='right')
            df=df.merge(price, on=['TradingDate', 'Stkcd'], how='right')
        elif freq=='M':  # 严格按照自然月进行调仓尝试
            price=self.price.groupby([pd.Grouper(key='TradingDate',freq='MS'),'Stkcd'])['price'].last().reset_index()
            factor= self.factor.groupby([pd.Grouper(key='TradingDate',freq='MS'),'Stkcd'])[self.fac_name].last().reset_index()
            Suspension_Limit = self.Suspension_Limit.groupby([pd.Grouper(key='TradingDate', freq='MS'), 'Stkcd'])[['Suspension', 'LimitStatus']].last().reset_index()
            df = factor.merge(Suspension_Limit, on=['TradingDate', 'Stkcd'], how='right')
            df = df.merge(price, on=['TradingDate', 'Stkcd'], how='right')
    
        # 计算收益率
        df['ret'] = df.groupby('Stkcd')['price'].pct_change()                   # 收益率
        df['ret'] = df.groupby('Stkcd')['ret'].shift(-1)                        # 下一期收益率
        df['ret'] = df.groupby('Stkcd')['ret'].fillna(0)                        # 收益率缺失的填充为0
        market_index = df.groupby('TradingDate')['ret'].mean().reset_index()
    
        TradingDates=df.groupby(df['TradingDate'])
        labels=['第1分位', '第2分位', '第3分位', '第4分位', '第5分位', '第6分位', '第7分位', '第8分位', '第9分位', '第10分位']  
        net_value=pd.DataFrame(index=['第1分位', '第2分位', '第3分位', '第4分位', '第5分位','第6分位', '第7分位', '第8分位', '第9分位', '第10分位',self.market_index_name,'Excess_+','Excess_-'])  # 每组的累计收益率
        group_ret = pd.DataFrame(index=['第1分位', '第2分位', '第3分位', '第4分位', '第5分位','第6分位', '第7分位', '第8分位', '第9分位', '第10分位',self.market_index_name,'Excess_+','Excess_-'])  # 每组的累计收益率
        last_date = df['TradingDate'].max()
        net_value['first'] = 1-0.0001-0.00001
        previousday_df = None

        for index,today_df in tqdm(TradingDates):           
            if index >= last_date:         # 跳过最后一期
                break

            today_df['quantile'] = today_df[self.fac_name].rank(method='first', pct=True)  # 对因子值进行排名

            try:
                today_df['quantile_group'] = pd.qcut(today_df['quantile'], q=10, labels=labels)
            except Exception as e:
                print(f"Error occurred at {index}: {e}") 
                continue
            
            # 处理停牌和涨跌停股票：将其调整为上周所在分组
            if previousday_df is not None:  
                today_df = today_df.merge(previousday_df,how='outer')
                mask = (today_df['Suspension'] == 1) | (today_df['LimitStatus'] == 1) | (today_df['LimitStatus'] == -1) 
                today_df.loc[mask, 'quantile_group'] = today_df.loc[mask, 'previous_group']
            else:
                today_df = today_df[(today_df['Suspension'] == 0)&(today_df['LimitStatus'] == 0)]

            r_mean=pd.DataFrame(index=labels).join(today_df.groupby('quantile_group')['ret'].mean())

            # 计算每个分组的交易成本
            if previousday_df is not None:
                for q in labels:
                    group_q = today_df[today_df['quantile_group'] == q]
                    group_q['change'] = (group_q['quantile_group'] != group_q['previous_group'])
                    transaction_cost = group_q['change'].sum()/len(group_q) * (0.001+2*0.0001+2*0.00001)
                    r_mean.loc[q] -= transaction_cost

            try:
                r_mean.loc[self.market_index_name]=market_index[market_index['TradingDate']==index]['ret'].iloc[0]
            except:
                r_mean.loc[self.market_index_name]=0  # 没有对应的数据就认为没有发生变化

            r_mean.loc['Excess_+']=r_mean.loc['第10分位']-r_mean.loc[self.market_index_name]
            r_mean.loc['Excess_-']=r_mean.loc['第1分位']-r_mean.loc[self.market_index_name]

            r_mean = r_mean.rename(columns={'ret': index})
            group_ret = group_ret.join(r_mean,how='left')
            net_value=net_value.join((1+r_mean).apply(lambda x: x*(net_value.iloc[:, -1])),how='left')   # 列表示对应的收益日期,根据收益率计算净值

            previousday_df = today_df[['Stkcd','quantile_group']]
            previousday_df.rename(columns={'quantile_group': 'previous_group'}, inplace=True)

        column_name = net_value.columns.to_list()
        del column_name[0]
        column_name.append(last_date)
        net_value.columns = column_name

        net_value=net_value.T
        if net_value['第10分位'].mean()>net_value['第1分位'].mean():    # 说明是正向因子
            net_value['Excess']=net_value['Excess_+']
        else:
            net_value['Excess']=net_value['Excess_-']

        net_value=net_value.drop(columns=['Excess_+','Excess_-'])
        colors = plt.cm.tab20.colors[:12]
        ax = net_value.plot(title=f'{factor_name}_{freq}_分层回测净值曲线', color=colors)
        if period == 'all':
            Bear_market = [('2015-06-12', '2016-01-27'),('2018-01-29', '2019-01-04'),('2021-12-13', '2024-09-20')]
            for bear_period in Bear_market:
                ax.axvspan(pd.to_datetime(bear_period[0]), pd.to_datetime(bear_period[1]), color='gray', alpha=0.3)
        ax.legend(loc='upper left')

        if save:
            if self.neutralize == None:
                net_value.T.to_csv(f"{self.dir}/{period}_无中性化_{freq}_1800主板_分组净值.csv")
                group_ret.to_csv(f"{self.dir}/{period}_无中性化_{freq}_1800主板_分组收益率.csv")
                plt.savefig(f"{self.dir}/{period}_无中性化_{freq}_1800主板_分组净值.png",dpi=250, bbox_inches='tight')
            if self.neutralize == 'cap':
                net_value.T.to_csv(f"{self.dir}/{period}_市值中性化_{freq}_1800主板_分组净值.csv")
                group_ret.to_csv(f"{self.dir}/{period}_市值中性化_{freq}_1800主板_分组收益率.csv")
                plt.savefig(f"{self.dir}/{period}_市值中性化_{freq}_1800主板_分组净值.png",dpi=250, bbox_inches='tight')
        self.net_value=net_value.T
        return net_value.T

    def aly_group_test(self,freq='W',save=False,risk_free_rate=0.02,period='all'):                         #     分析分组回测的结果
        df=self.net_value.T
        pct=df.pct_change().dropna(how='any')

        # 单调性结果分析
        test = pd.DataFrame({'分层累计收益率': df.T.iloc[:10, -1], 'Group': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        print("单调性分析：", test.corr().iloc[0, 1])

        # 设置时间频率为52（默认为周频）
        times = 52
        if freq == 'M':
            times = 12
        elif freq == 'D':
            times = 252            
        
        # 计算年化标准差、年化收益率、夏普比率和最大回撤
        groupStd = pct.std() * np.sqrt(times)
        temp = df.iloc[-1, :] / df.iloc[0, :]
        groupYield = (np.power(np.absolute(temp), times / len(pct)) - 1) * np.sign(temp)  # 几何平均年化收益率
        
        # 计算夏普比率
        groupSharp = (groupYield - risk_free_rate) / groupStd


        def MaxDrawdown(return_list):
            return ((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list)).max()

        groupMaxDrawdown = df.apply(MaxDrawdown, axis=0)
        
        # 计算累计收益率
        groupCumulativeReturn = df.iloc[-1, :] / df.iloc[0, :] - 1
    
        # 汇总分析结果
        result = pd.concat([groupCumulativeReturn, groupYield, groupStd, groupSharp, groupMaxDrawdown], axis=1)
        result.columns = ['CumulativeReturn', 'anlYield', 'anlStd', 'Sharp', 'MaxDrawdown']    

        print(result)

        # 如果需要保存结果
        if save:
            result['单调性'] = test.corr().iloc[0, 1]  # 方便记录
            if self.neutralize == None:
                result.to_csv(f"{self.dir}/{period}_无中性化_{freq}_1800主板_分组回测结果.csv")
            if self.neutralize == 'cap':
                result.to_csv(f"{self.dir}/{period}_市值中性化__{freq}_1800主板_分组回测结果.csv")
    
    
if __name__=="__main__":
    alpha_path="/Users/xuyanye/Desktop/quant_mm/factor/alpha" # 您的因子文件目录
    result_path=f"result" # 结果保存目录

    # 在回测函数外只读取不处理不填充数据！！！
    # 请根据您的实际路径修改以下数据文件路径
    close_price=pd.read_parquet(f"/Users/xuyanye/Desktop/quant_mm/adj_close.parquet", engine='pyarrow')
    Suspension_Limit = pd.read_parquet(f"/Users/xuyanye/Desktop/quant_mm/Suspension_Limit.parquet", engine='pyarrow') 
    stock_pool = pd.read_parquet(f"/Users/xuyanye/Desktop/quant_mm/history-1800.pq", engine='pyarrow') 

    market_value=pd.read_parquet(f"/Users/xuyanye/Desktop/quant_mm/mv_turn.parquet", engine='pyarrow')[['TradingDate','Stkcd','MarketValue']]
    market_value['TradingDate']=pd.to_datetime(market_value['TradingDate'])

    factors=os.listdir(alpha_path)
    market_index='1800' 

    for factor_file in factors:
        # 首先定义 factor_name
        if factor_file.endswith('csv'):
            factor_name = factor_file[:-4]
        elif factor_file.endswith('parquet'):
            factor_name = factor_file[:-8]
        elif factor_file.endswith('pq'):
            factor_name = factor_file[:-3]
        else:
            print(f"跳过不支持的文件格式: {factor_file}")
            continue
            
        print(f"--- Processing Factor: {factor_name} ---") # 输出英文
        
        # 然后根据文件类型读取数据
        if factor_file.endswith('.csv'):
            df_factor = pd.read_csv(os.path.join(alpha_path, factor_file))
        elif factor_file.endswith('.parquet') or factor_file.endswith('.pq'):
            df_factor = pd.read_parquet(os.path.join(alpha_path, factor_file), engine='pyarrow')
        
        result = f"{result_path}/{factor_name}"
        print(factor_name)


        current_result_dir = os.path.join(result_path, factor_name) # 为每个因子创建单独的结果目录
        if not os.path.exists(current_result_dir):
            os.makedirs(current_result_dir)

        # --- 无中性化回测 (2010-2025) ---
        print(f"--- Running {factor_name} No Neutralization Backtest (2010-2025) ---") # 输出英文
        test_no_neutral = back_test(df_factor.copy(), close_price.copy(), factor_name, current_result_dir, market_index, Suspension_Limit.copy(), start_date='2010-01-01', end_date='2025-06-30')
        test_no_neutral.factor_process(stock_pool=stock_pool.copy(), neutralize=None) # 无中性化
        test_no_neutral.group_test(freq='W', save=True, period='all') # 保存净值曲线图 (英文)
        test_no_neutral.aly_group_test(freq='W', save=True, period='all') # 保存并打印指标 (英文)

        # --- 市值中性化回测 (2010-2025) ---
        print(f"--- Running {factor_name} Market Cap Neutralization Backtest (2010-2025) ---") # 输出英文
        test_cap_neutral = back_test(df_factor.copy(), close_price.copy(), factor_name, current_result_dir, market_index, Suspension_Limit.copy(), start_date='2010-01-01', end_date='2025-06-30')
        test_cap_neutral.factor_process(stock_pool=stock_pool.copy(), market_value=market_value.copy(), neutralize='cap') # 市值中性化
        test_cap_neutral.group_test(freq='W', save=True, period='all') # 保存净值曲线图 (英文)
        test_cap_neutral.aly_group_test(freq='W', save=True, period='all') # 保存并打印指标 (英文)

        print(f"--- {factor_name} Factor Backtest Completed ---\n") # 输出英文