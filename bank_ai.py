import pandas as pd
import matplotlib as plt
df=pd.read_csv("https://raw.githubusercontent.com/juliencohensolal/BankMarketing/master/rawData/bank-additional-full.csv",sep=';')

no_count=0
yes_count=0
not_filtered=0
for i in df["y"]:
    if (i=="no") or (i=="NO"):
        no_count=no_count+1
    elif(i=="yes") or (i=="YES"):
        yes_count=yes_count+1
    else:
        not_filtered=not_filtered+1

print(str(no_count)+"    no_count")
print(str(yes_count)+"   yes_count")