"""
trainer/train_from_file.py

Helper to run end-to-end training from a dataset file (CSV or XLSX).
"""
import pandas as pd
from features.engineer import FeatureEngineer
from trainer.pipeline import LSTMModelTrainer


def train_from_file(path, sequence_length=9):
    if path.endswith('.xlsx') or path.endswith('.xls'):
        df = pd.read_excel(path, engine='openpyxl')
    else:
        df = pd.read_csv(path)

    df = df.sort_values(by='ExecutionDate')

    # Feature engineering to prepare price, indicator, time features in a pandas DataFrame
    fe = FeatureEngineer()
    df = fe.run(df)

    price_features = ['Entry_vs_PrevClose', 'EntryPriceChange', 'volatility']
    indicator_features = ['EMA_10', 'EMA_20', 'MA50', 'BB_Width', 'RSI', 'Momentum', 'ATR']
    time_features = ['HourOfDay', 'OrderMonth', 'GoldenCrossover']

    trainer = LSTMModelTrainer(sequence_length=sequence_length)

    Xp, Xi, Xt, y = trainer.preprocess(df, price_features, indicator_features, time_features)

    trainer.build_model(price_dim=len(price_features), indicator_dim=len(indicator_features), time_dim=len(time_features))
    trainer.compile()

    trainer.fit(Xp, Xi, Xt, y)

    val_size = int(len(y) * 0.3)
    trainer.evaluate(Xp[-val_size:], Xi[-val_size:], Xt[-val_size:], y[-val_size:])

    trainer.save()
    return True
