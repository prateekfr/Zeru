
import os, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    # 1. Setup directories
    raw_dir = Path('data/raw')
    proc_dir = Path('data/processed')
    feat_dir = Path('data/features')
    out_dir = Path('outputs')
    for d in (proc_dir, feat_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load & flatten all JSON chunks
    dfs = []
    for path in sorted(raw_dir.glob('compoundV2_transactions_ethereum_chunk_*.json')):
        print(f"Loading {path.name}…")
        data = json.load(open(path, 'r'))
        for action, recs in data.items():
            df = pd.json_normalize(recs)
            df['action'] = action
            dfs.append(df)
    txns = pd.concat(dfs, ignore_index=True)
    txns.rename(columns={
        'account.id': 'wallet_id',
        'asset.symbol': 'asset_symbol',
        'hash': 'tx_hash'
    }, inplace=True)
    txns['timestamp'] = pd.to_datetime(txns['timestamp'].astype(int), unit='s')
    txns.to_parquet(proc_dir/'transactions.parquet', index=False)
    print(f"✅ Flattened {len(txns):,} txns")

    # 3. Clean & filter to core actions
    core_actions = ['deposits','borrows','repays','withdraws','liquidates']
    core = txns[txns['action'].isin(core_actions)].copy()
    core['amountUSD'] = core['amountUSD'].astype(float)
    core.dropna(subset=['wallet_id','timestamp'], inplace=True)
    core.to_parquet(proc_dir/'transactions_clean.parquet', index=False)
    print(f"✅ Kept {len(core):,} core actions")

    # 4. Feature engineering
    agg = core.groupby('wallet_id').agg(
        txn_count=('action','count'),
        total_usd=('amountUSD','sum'),
        repays=('action', lambda x: (x=='repays').sum()),
        borrows=('action', lambda x: (x=='borrows').sum()),
        liquidates=('action', lambda x: (x=='liquidates').sum()),
        first_ts=('timestamp','min'),
        last_ts=('timestamp','max'),
        unique_assets=('asset_symbol','nunique')
    ).reset_index()

    # Add derived features
    agg['active_days'] = (agg['last_ts'] - agg['first_ts']).dt.days + 1
    agg['txns_per_day'] = agg['txn_count'] / agg['active_days'].replace(0,1)
    agg['borrow_to_repay'] = agg['borrows'] / agg['repays'].replace(0,1)
    agg['avg_txn_value_usd'] = agg['total_usd'] / agg['txn_count'].replace(0,1)
    agg['liq_rate'] = agg['liquidates'] / agg['txn_count'].replace(0,1)
    agg['repay_rate'] = agg['repays'] / agg['borrows'].replace(0,1)
    agg['wallet_lifetime_days'] = agg['active_days']
    agg.drop(columns=['first_ts','last_ts'], inplace=True)

    # 5. Rule-based scoring
    agg['usd_pct'] = agg['total_usd'].rank(pct=True)
    agg['txn_pct'] = agg['txn_count'].rank(pct=True)
    agg['repay_pct'] = agg['repays'].rank(pct=True)
    agg['liq_pct'] = agg['liquidates'].rank(pct=True)
    agg['activity_pct'] = (agg['usd_pct'] + agg['txn_pct'] + agg['repay_pct']) / 3
    agg['final_score'] = (0.7*agg['activity_pct'] + 0.3*(1-agg['liq_pct']))*100
    fs_min, fs_max = agg['final_score'].min(), agg['final_score'].max()
    agg['final_score'] = ((agg['final_score'] - fs_min) / (fs_max - fs_min)) * 100
    agg['final_score'] = agg['final_score'].round(2)

    # 5b. ML-based clustering score
    features_ml = ['txn_count', 'total_usd', 'repays', 'liquidates', 'txns_per_day', 'unique_assets']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg[features_ml])
    kmeans = KMeans(n_clusters=4, random_state=42)
    agg['cluster'] = kmeans.fit_predict(X_scaled)
    cluster_means = agg.groupby('cluster')['final_score'].mean().sort_values(ascending=False)
    cluster_to_score = {cluster: i for i, cluster in enumerate(cluster_means.index[::-1])}
    agg['ml_score'] = agg['cluster'].map(cluster_to_score).astype(float)
    agg['ml_score'] = (agg['ml_score'] / agg['ml_score'].max()) * 100

    # 6. Blended score 
    agg['blended_score'] = (0.6 * agg['final_score']) + (0.4 * agg['ml_score'])

    # 7. Save features and scores
    agg.to_parquet(feat_dir/'wallet_features_ml.parquet', index=False)
    agg.to_csv(feat_dir/'wallet_features_ml.csv', index=False)
    top1000 = agg[['wallet_id','blended_score']]                 .sort_values('blended_score', ascending=False)                 .head(1000)
    top1000.to_csv(out_dir / 'top_1000_wallet_scores_ml.csv', index=False)
    print(f"✅ Top 1,000 scores saved to {out_dir/'top_1000_wallet_scores_ml.csv'}")

if __name__ == '__main__':
    main()
