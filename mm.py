import pandas as pd
import re
mm=pd.read_csv("scam_dataset.csv")
mm["message"]=mm["message"].str.lower()
mm["message"]=mm["message"].apply (lambda x:  re.sub(r"[^a-z\s]", "", x))

print(mm["message"].head