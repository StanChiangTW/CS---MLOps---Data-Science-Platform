from src.dsba.data_ingestion.data_ingestion import load_csv

data = load_csv('data/BankChurners.csv')
print(data.head())
