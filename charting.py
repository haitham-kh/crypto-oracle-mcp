import os
import pandas as pd
import mplfinance as mpf

def generate_chart(candles, sr_data, symbol):
    """Generate a candlestick chart with VPVR Point of Control and S/R lines."""
    try:
        if not candles: return None
        df = pd.DataFrame(candles)
        if 'close_time' in df.columns:
            df['Date'] = pd.to_datetime(df['close_time'], unit='ms')
        elif 'open_time' in df.columns:
            df['Date'] = pd.to_datetime(df['open_time'], unit='ms')
        else:
            df['Date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='H')
            
        df.set_index('Date', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        df = df.tail(150) # Show last 150 candles
        
        lines = []
        colors = []
        
        poc = sr_data.get('vpvr_poc')
        if poc and min(df['low']) * 0.9 < poc < max(df['high']) * 1.1:
            lines.append(poc)
            colors.append('magenta')
            
        for s in sr_data.get('strong_supports', [])[:2]:
            lvl = s['level']
            if min(df['low']) * 0.9 < lvl < max(df['high']) * 1.1:
                lines.append(lvl)
                colors.append('g')
                
        for r in sr_data.get('strong_resistances', [])[:2]:
            lvl = r['level']
            if min(df['low']) * 0.9 < lvl < max(df['high']) * 1.1:
                lines.append(lvl)
                colors.append('r')
                
        os.makedirs('charts', exist_ok=True)
        save_path = f"charts/{symbol}_chart.png"
        
        # Determine max and min of y-axis to safely plot lines
        valid_lines = []
        valid_colors = []
        for l, c in zip(lines, colors):
            if l > df['low'].min() * 0.8 and l < df['high'].max() * 1.2:
                valid_lines.append(l)
                valid_colors.append(c)
                
        if valid_lines:
            mpf.plot(df, type='candle', volume=True, style='charles',
                     hlines=dict(hlines=valid_lines, colors=valid_colors, linestyle='dashed'),
                     title=f"{symbol} Price Action & VPVR S/R",
                     savefig=save_path)
        else:
            mpf.plot(df, type='candle', volume=True, style='charles',
                     title=f"{symbol} Price Action",
                     savefig=save_path)
        
        return os.path.abspath(save_path)
    except Exception as e:
        print("Charting error:", str(e))
        return None
