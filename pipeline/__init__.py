"""
ESN data pipeline.

Modules:
    fetch     NSE + Yahoo Finance raw data download
    silver    Raw → clean parquet silver tables
    universe  Training vs active universe management
    labels    Build t+5 max/min binary labels (6 columns)
    zones     Support / resistance zone detection
    features  Build gold/features.parquet (~90 features)
    train     Walk-forward CV training of 6 LightGBM classifiers
    predict   Daily inference + API helpers
    evaluate  Backtest, lift, calibration, IC reporting
"""
