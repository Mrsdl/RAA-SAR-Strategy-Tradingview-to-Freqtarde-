import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import reduce

import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.strategy.parameters import CategoricalParameter
from pandas import DataFrame
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class FRAMAIndicator:
    """FRAMA (Fractal Adaptive Moving Average) 指标计算类"""
    
    def __init__(self, length: int = 50):
        self.length = length
        
    def calculate_frama(self, dataframe: DataFrame) -> pd.Series:
        """计算FRAMA指标"""
        df = dataframe.copy()
        high = df['high']
        low = df['low']
        close = df['close']
        
        halflen = self.length // 2
        frama = pd.Series(index=close.index, dtype=np.float64)
        dseries = pd.Series(index=close.index, dtype=np.float64)
        
        # 初始化
        frama.iloc[self.length-1] = close.iloc[self.length-1]
        dseries.iloc[self.length-1] = 1.0
        
        for i in range(self.length, len(close)):
            # 计算分形维度
            hi1 = high.iloc[i - self.length + halflen + 1: i + 1].max()
            lo1 = low.iloc[i - self.length + halflen + 1: i + 1].min()
            hi2 = high.iloc[i - self.length + 1: i - halflen + 1].max()
            lo2 = low.iloc[i - self.length + 1: i - halflen + 1].min()
            hi3 = high.iloc[i - self.length + 1: i + 1].max()
            lo3 = low.iloc[i - self.length + 1: i + 1].min()
            
            hl1 = (hi1 - lo1) / halflen
            hl2 = (hi2 - lo2) / halflen
            hl = (hi3 - lo3) / self.length
            
            if hl1 > 0 and hl2 > 0 and hl > 0:
                d = (np.log((hl1 + hl2) / hl)) / np.log(2)
            else:
                d = dseries.iloc[i-1] if i > 0 else 1.0
            
            dseries.iloc[i] = d
            
            # 计算自适应系数
            alpha = max(0.01, min(1, np.exp(-4.6 * (d - 1))))
            
            # 计算FRAMA
            frama.iloc[i] = (1 - alpha) * frama.iloc[i - 1] + alpha * close.iloc[i]
            
        return frama


class RAAIndicator:
    """RAA (Relative Average Absolute) 指标计算类 - 基于TradingView策略"""
    
    def __init__(self, length: int = 50, raa_mult: float = 1.35):
        self.length = length
        self.raa_mult = raa_mult
        self.frama_indicator = FRAMAIndicator(length)
        
    def calculate_raa(self, dataframe: DataFrame) -> DataFrame:
        """计算RAA指标"""
        df = dataframe.copy()
        close = df['close']
        
        # 计算FRAMA
        frama = self.frama_indicator.calculate_frama(df)
        
        # 计算RAA值（平均绝对偏差）
        abs_deviations = np.abs(close - frama)
        raa_value = abs_deviations.rolling(window=self.length).mean()
        
        # 计算上下轨
        upper_band = frama + (raa_value * self.raa_mult)
        lower_band = frama - (raa_value * self.raa_mult)
        
        # 计算趋势方向
        direction = pd.Series(index=close.index, dtype=int)
        direction.iloc[self.length-1] = 0
        
        for i in range(self.length, len(close)):
            if (close.iloc[i] > upper_band.iloc[i] and 
                close.iloc[i-1] <= upper_band.iloc[i-1]):
                direction.iloc[i] = 1
            elif (close.iloc[i] < lower_band.iloc[i] and 
                  close.iloc[i-1] >= lower_band.iloc[i-1]):
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 0
        
        df['raa_frama'] = frama
        df['raa_upper'] = upper_band
        df['raa_lower'] = lower_band
        df['raa_value'] = raa_value
        df['raa_direction'] = direction
        
        return df


class SARIndicator:
    """SAR (Parabolic SAR) 指标计算类 - 支持双SAR"""
    
    def __init__(self, af: float = 0.06, af_increase: float = 0.02, max_af: float = 0.2):
        self.af = af
        self.af_increase = af_increase
        self.max_af = max_af
        
    def calculate_sar(self, dataframe: DataFrame) -> DataFrame:
        """计算SAR指标"""
        df = dataframe.copy()
        low = df['low']
        high = df['high']
        close = df['close']
        
        trend = pd.Series(index=df.index, dtype=float)
        ep = pd.Series(index=df.index, dtype=float)
        sar = pd.Series(index=df.index, dtype=float)
        af = pd.Series(index=df.index, dtype=float)
        
        # 初始化
        if close.iloc[1] > close.iloc[0]:
            trend.iloc[1] = 1
            sar.iloc[1] = low.iloc[1]
            ep.iloc[1] = high.iloc[1]
        else:
            trend.iloc[1] = -1
            sar.iloc[1] = high.iloc[1]
            ep.iloc[1] = low.iloc[1]
        af.iloc[1] = self.af
        
        # 计算SAR
        for i in range(2, len(df)):
            if trend.iloc[i-1] == 1:  # 上涨趋势
                ep.iloc[i] = max(ep.iloc[i-1], high.iloc[i])
                if ep.iloc[i] == ep.iloc[i-1]:
                    af.iloc[i] = af.iloc[i-1]
                else:
                    af.iloc[i] = min(af.iloc[i-1] + self.af_increase, self.max_af)
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2])
                trend.iloc[i] = 1
                if low.iloc[i] < sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = self.af
                    ep.iloc[i] = low.iloc[i]
            elif trend.iloc[i-1] == -1:  # 下跌趋势
                ep.iloc[i] = min(ep.iloc[i-1], low.iloc[i])
                if ep.iloc[i] == ep.iloc[i-1]:
                    af.iloc[i] = af.iloc[i-1]
                else:
                    af.iloc[i] = min(af.iloc[i-1] + self.af_increase, self.max_af)
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2])
                trend.iloc[i] = -1
                if high.iloc[i] > sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = self.af
                    ep.iloc[i] = high.iloc[i]
        
        df['sar_value'] = sar
        df['sar_trend'] = trend
        df['sar_ep'] = ep
        df['sar_af'] = af
        df['sar_reverse'] = trend.diff()
        
        return df


class RAA_SAR_Strategy(IStrategy):
    """
    RAA-SAR策略 - 基于TradingView Pine Script实现
    
    策略逻辑:
    1. 使用RAA指标识别入场时机和波动确认
    2. 使用双SAR指标进行趋势跟踪和止损保护
    3. 当RAA上穿信号点且RAA值下降时开多仓
    4. 当SAR反转或时间间隔到达时平仓
    5. 设置止盈止损保护
    """
    
    # 策略参数
    INTERFACE_VERSION = 3
    
    # 绘图配置
    plot_config = {
        'main_plot': {
            'raa_frama': {'color': 'blue', 'type': 'scatter', 'mode': 'lines'},
            'raa_upper': {'color': 'red', 'type': 'scatter', 'mode': 'lines'},
            'raa_lower': {'color': 'green', 'type': 'scatter', 'mode': 'lines'},
            'sar_value': {'color': 'purple', 'type': 'scatter'},
            'sar_long_value': {'color': 'orange', 'type': 'scatter'},
        },
        'subplots': {
            "RAA指标": {
                'raa_value': {'color': 'orange'},
                'raa_direction': {'color': 'brown'},
            },
            "SAR指标": {
                'sar_trend': {'color': 'blue'},
                'sar_long_trend': {'color': 'orange'},
            },
            "信号": {
                'raa_signal': {'color': 'green'},
                'sar_reverse': {'color': 'red'},
            }
        }
    }
    
    # 最小ROI表 - 基于TradingView策略的止盈设置
    minimal_roi = {
        "0": 0.05,      # 5%止盈
        "60": 0.03,     # 1小时内3%止盈
        "120": 0.02,    # 2小时内2%止盈
        "240": 0.01,    # 4小时内1%止盈
    }
    
    # 止损设置 - 基于TradingView策略的止损设置
    stoploss = -0.025   # 2.5%止损
    
    # 时间框架
    timeframe = '1h'
    
    # 参数优化 - 基于TradingView策略的默认参数
    # RAA参数
    raa_length = IntParameter(30, 70, default=50, space="buy")
    raa_mult = DecimalParameter(1.0, 1.5, default=1.35, space="buy", decimals=2)
    
    # SAR参数
    sar_af = DecimalParameter(0.01, 0.1, default=0.06, space="buy", decimals=3)
    sar_af_increase = DecimalParameter(0.01, 0.05, default=0.02, space="buy", decimals=3)
    sar_max_af = DecimalParameter(0.1, 0.3, default=0.2, space="buy", decimals=1)
    
    # 长线SAR参数
    sar_long_af = DecimalParameter(0.001, 0.01, default=0.001, space="buy", decimals=4)
    sar_long_af_increase = DecimalParameter(0.001, 0.01, default=0.001, space="buy", decimals=4)
    sar_long_max_af = DecimalParameter(0.1, 0.3, default=0.2, space="buy", decimals=1)
    
    # 时间间隔参数
    gap_time = IntParameter(6, 24, default=12, space="buy")  # 小时
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算技术指标
        """
        # 计算RAA指标
        raa_indicator = RAAIndicator(
            length=self.raa_length.value,
            raa_mult=self.raa_mult.value
        )
        dataframe = raa_indicator.calculate_raa(dataframe)
        
        # 计算普通SAR指标
        sar_indicator = SARIndicator(
            af=self.sar_af.value,
            af_increase=self.sar_af_increase.value,
            max_af=self.sar_max_af.value
        )
        dataframe = sar_indicator.calculate_sar(dataframe)
        
        # 计算长线SAR指标
        sar_long_indicator = SARIndicator(
            af=self.sar_long_af.value,
            af_increase=self.sar_long_af_increase.value,
            max_af=self.sar_long_max_af.value
        )
        dataframe = sar_long_indicator.calculate_sar(dataframe)
        
        # 重命名长线SAR列
        dataframe['sar_long_value'] = dataframe['sar_value']
        dataframe['sar_long_trend'] = dataframe['sar_trend']
        dataframe['sar_long_ep'] = dataframe['sar_ep']
        dataframe['sar_long_af'] = dataframe['sar_af']
        dataframe['sar_long_reverse'] = dataframe['sar_reverse']
        
        # 重新计算普通SAR（覆盖长线SAR的计算结果）
        dataframe = sar_indicator.calculate_sar(dataframe)
        
        # 计算RAA信号 - 基于TradingView策略逻辑
        dataframe['raa_signal'] = 0
        for i in range(2, len(dataframe)):
            # 连续3个方向信号为1的条件
            direction_sum = (dataframe['raa_direction'].iloc[i] + 
                           dataframe['raa_direction'].iloc[i-1] + 
                           dataframe['raa_direction'].iloc[i-2])
            
            # RAA值下降条件（当前RAA值小于前一个或前两个RAA值）
            raa_decline = ((dataframe['raa_value'].iloc[i] < dataframe['raa_value'].iloc[i-1]) or
                          (dataframe['raa_value'].iloc[i-1] < dataframe['raa_value'].iloc[i-2]))
            
            # 综合条件：方向信号为1且RAA值下降
            if direction_sum == 1 and raa_decline:
                dataframe.loc[dataframe.index[i], 'raa_signal'] = 1
        
        # 计算RAA值变化（保留原有逻辑用于其他用途）
        dataframe['raa_decline'] = (
            (dataframe['raa_value'] < dataframe['raa_value'].shift(1)) |
            (dataframe['raa_value'].shift(1) < dataframe['raa_value'].shift(2))
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成买入信号 - 基于TradingView策略逻辑
        """
        # RAA信号条件（已经包含了RAA值下降的检查）
        raa_signal_up = dataframe['raa_signal'] == 1
        
        # 综合买入条件 - 与Pine Script逻辑一致
        dataframe.loc[
            raa_signal_up,
            'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成卖出信号 - 基于TradingView策略逻辑
        """
        # SAR反转信号
        sar_reverse = dataframe['sar_reverse'] == -2
        
        # 长线SAR反转信号
        sar_long_reverse = dataframe['sar_long_reverse'] == -2
        
        # 综合卖出条件
        dataframe.loc[
            sar_reverse | sar_long_reverse,
            'exit_long'] = 1
        
        return dataframe
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[str]:
        """
        自定义退出逻辑 - 基于TradingView策略的时间间隔控制
        """
        # 计算持仓时间（小时）
        hold_time = (current_time - trade.open_date_utc).total_seconds() / 3600
        
        # 时间间隔控制 - 基于TradingView策略的gap_time逻辑
        if hold_time >= self.gap_time.value:
            # 检查长线SAR和RAA条件
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            
            # 如果RAA值下降且长线SAR在低点下方，开空仓（这里我们只平多仓）
            if (last_candle['raa_value'] < dataframe['raa_value'].iloc[-2] and 
                last_candle['sar_long_value'] < last_candle['low']):
                return "time_gap_raa_sar"
        
        # 1.5倍时间间隔的SAR反转退出
        if hold_time >= 1.5 * self.gap_time.value:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            
            # SAR反转且趋势不为上涨
            if (last_candle['sar_value'] > last_candle['low'] and 
                last_candle['sar_value'] < last_candle['high'] and
                last_candle['sar_value'] > dataframe['sar_value'].iloc[-2] and
                last_candle['raa_direction'] != 1):
                return "sar_reverse_1_5x_gap"
        
        return None
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        自定义止损逻辑 - 基于TradingView策略的止盈止损设置
        """
        # 基于TradingView策略的止盈止损设置
        # 止盈5%，止损2.5%
        if current_profit >= 0.05:  # 5%止盈
            return -0.001  # 保护利润
        elif current_profit <= -0.025:  # 2.5%止损
            return -0.001  # 止损
        
        return self.stoploss
