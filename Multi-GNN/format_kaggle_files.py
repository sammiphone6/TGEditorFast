import pandas as pd
from datetime import datetime
import sys
import os

n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = sys.argv[1]
folder_path = sys.argv[1].split('.')[0]
outPath = folder_path + "/formatted_transactions.csv"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

raw = pd.read_csv(inPath, dtype=str, keep_default_na=False)

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format, Embedding, Is Laundering\n"

firstTs = -1

data = []
for i, row in raw.iterrows(): #HAD PROBLEMS WITH LAST ROW SO SOLVED THIS WAY
    datetime_object = datetime.strptime(row["Timestamp"], '%Y/%m/%d %H:%M')
    ts = datetime_object.timestamp()
    day = datetime_object.day
    month = datetime_object.month
    year = datetime_object.year
    hour = datetime_object.hour
    minute = datetime_object.minute

    if firstTs == -1:
        startTime = datetime(year, month, day)
        firstTs = startTime.timestamp() - 10

    ts = ts - firstTs

    cur1 = get_dict_val(row["Receiving Currency"], currency)
    cur2 = get_dict_val(row["Payment Currency"], currency)

    fmt = get_dict_val(row["Payment Format"], paymentFormat)

    fromAccIdStr = row["From Bank"] + row[2]
    fromId = get_dict_val(fromAccIdStr, account)

    toAccIdStr = row["To Bank"] + row[4]
    toId = get_dict_val(toAccIdStr, account)

    amountReceivedOrig = float(row["Amount Received"])
    amountPaidOrig = float(row["Amount Paid"])

    # emb = row["Embedding"]

    isl = int(row["Is Laundering"])

    data.append([i, fromId, toId, ts, amountPaidOrig, cur2, amountReceivedOrig, cur1, fmt, isl])
    # data.append([i, fromId, toId, ts, amountPaidOrig, cur2, amountReceivedOrig, cur1, fmt, emb, isl])

formatted = pd.DataFrame(data, columns=['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent', 'Sent Currency', 'Amount Received', 'Received Currency', 'Payment Format', 
                                        # 'Embedding', 
                                        'Is Laundering'])
# formatted.sort_values(by=['Timestamp'], inplace=True)

formatted.to_csv(outPath, index=False)