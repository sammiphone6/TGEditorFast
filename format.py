import argparse
import pandas as pd
from datetime import datetime
import time
import torch
import sys
import os
from datetime import datetime

def preprocess(args):
    print('Preprocessing Data')
    start = time.time()

    min_max_times = {
        "Chase": ['2022-09-01 00:00', '2022-09-18 16:18'], #just anything here is fine
        "HI-Tiny": ['2022-09-01 00:00', '2022-09-18 16:18'],
        "HI-Small": ['2022-09-01 00:00', '2022-09-18 16:18'],
        "HI-Medium": ['2022-09-01 00:00', '2022-09-18 16:18'],
        "HI-Large": ['2022-09-01 00:00', '2022-09-18 16:18'],
        "LI-Tiny": ['2022-09-01 00:00', '2022-09-17 00:00'],
        "LI-Small": ['2022-09-01 00:00', '2022-09-17 00:00'],
        "LI-Medium": ['2022-09-01 00:00', '2022-09-17 00:00'],
        "LI-Large": ['2022-09-01 00:00', '2022-09-17 00:00'],
    }

    df = pd.read_csv(args.in_path, dtype=str, keep_default_na=False)
    df['t'] = df['t'].astype(float)

    # df.sort_values(by='t', inplace=True)
    max_t = df['t'].max()

    def convert(x):
        # Convert to Unix timestamp
        if args.task not in min_max_times:
            print("ARGS.TASK NOT IN MIN_MAX_TIMES, USING DEFAULT VALUE, FORMAT.PY LINE 35")
            min_max_times[args.task] = ['2022-09-01 00:00', '2022-09-18 16:18']
        dt1 = datetime.strptime(min_max_times[args.task][0], '%Y-%m-%d %H:%M')
        dt2 = datetime.strptime(min_max_times[args.task][1], '%Y-%m-%d %H:%M')
        unix_1 = int(dt1.timestamp())
        unix_2 = int(dt2.timestamp())

        # if p:
        #     unix_new = unix_1 + (unix_2 - unix_1) * float(x) / max_t
        # else:
        unix_new = unix_1 + (unix_2 - unix_1) * float(x / max_t)
        
        # Convert Unix timestamp back to the original format
        dt_new = datetime.fromtimestamp(round(unix_new)).strftime('%Y/%m/%d %H:%M')
        
        return dt_new

    df['t'] = (df['t']).map(lambda x: convert(x))#.astype(str)

    df_renamed = df.rename(columns={
        'src': 'From Bank',
        'tar': 'To Bank',
        't': 'Timestamp',
        'label': 'Is Laundering'
    })

    # print(time.time()-start)
    # print(len(df_renamed))

    # and need to create them with default/unknown values
    unknown_value = "1"

    columns_to_change = [
        'Account (placeholder)',
        'Account.1 (placeholder)',
        'Bank',
        'Amount Received',
        'Receiving Currency',
        'Amount Paid',
        'Payment Currency',
        'Payment Format',
    ]

    df_renamed[columns_to_change] = unknown_value
    desired_order = ['Timestamp', 'From Bank', 'Account (placeholder)', 'To Bank', 'Account.1 (placeholder)', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering']
    df_final = df_renamed[desired_order]

    return df_final


def run(args):

    folder_path = args.out_path_folder
    filename = args.out_path_filename
    outPath = f"{folder_path}/{filename}"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    raw = preprocess(args)
    print('Finished Preprocessing Data. Now formatting for Multi-GNN.')

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

    # header = "EdgeID,from_id,to_id,Timestamp,\
    # Amount Sent,Sent Currency,Amount Received,Received Currency,\
    # Payment Format, Is Laundering\n"

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

        isl = int(float(row["Is Laundering"]))

        data.append([i, fromId, toId, ts, amountPaidOrig, cur2, amountReceivedOrig, cur1, fmt, isl])

    formatted = pd.DataFrame(data, columns=['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent', 'Sent Currency', 'Amount Received', 'Received Currency', 'Payment Format', 'Is Laundering'])

    formatted.to_csv(outPath, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--in_path", type=str, default="", help="Path to input data")
    parser.add_argument("--out_path_folder", type=str, default="", help="Folder Path to write resulting data for Multi-GNN")
    parser.add_argument("--out_path_filename", type=str, default="", help="Filename to write resulting data for Multi-GNN")
    parser.add_argument("--task", type=str, default="None given", help="Task to run")
    args = parser.parse_args()

    run(args)