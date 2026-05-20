import pickle, datetime as dt
for s in ["LINKUSDT", "SOLUSDT", "BTCUSDT", "AVAXUSDT"]:
    with open(f"challenge_data/{s}_challenge.pkl", "rb") as f:
        d = pickle.load(f)
    first = dt.datetime.utcfromtimestamp(d[0]["timestamp"] / 1000)
    last = dt.datetime.utcfromtimestamp(d[-1]["timestamp"] / 1000)
    print(f"{s}: {len(d):,} candles  first={first}  last={last}")
