import pandas as pd

mail = pd.read_csv("scam_dataset.csv")
mail = mail["message"]
mail = [text.lower() for text in mail]

print(mail[:5])
