# Zeru Finance : Decentralized Wallet Scoring

This repo is the submission of POC for Zeru Finance by creating a credit scorer in which a machine learning model assigns a credit score between 0 and 100 to each wallet from raw, transaction-level data from the Compound V2 protocol.

## Folder Structure 

```
Zeru/
├── data/
│   ├── raw/                        # Raw Compound V2 JSON files (input)
│   ├── processed/                  # Flattened and cleaned data (Parquet)
│   └── features/                   # Wallet-level feature data (CSV/Parquet)
├── outputs/
│   ├── top_1000_wallet_scores_ml.csv  # Final 1000 scored wallets
├── reports/
│   ├── methodology.md / .pdf      # Methodology explanation
│   ├── wallet_analysis.md / .pdf  # Top 5 / Bottom 5 wallet behavior
├── compound_scoring_ml.py         # Main scoring pipeline (rule + ML)
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies
```

