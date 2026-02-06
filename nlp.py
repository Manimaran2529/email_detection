import pandas as pd
mani=pd.read_csv("scam_dataset.csv")
mani["message"]=mani["message"].str.lower()
