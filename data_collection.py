import os
import codecs
import pandas as pd
import chardet

def _to_utf8(filename: str, encoding="latin1", blocksize=1048576):
    tmpfilename = filename + ".tmp"
    with codecs.open(filename, "r", encoding) as source:
        with codecs.open(tmpfilename, "w", "utf-8") as target:
            while True:
                contents = source.read(blocksize)
                if not contents:
                    break
                target.write(contents)

    os.remove(filename)
    # replace the original file
    os.rename(tmpfilename, filename)

def _renaming_class_label(df: pd.DataFrame):
    labels = {"Web Attack \x96 Brute Force": "Web Attack-Brute Force",
              "Web Attack \x96 XSS": "Web Attack-XSS",
              "Web Attack \x96 Sql Injection": "Web Attack-Sql Injection"}

    for old_label, new_label in labels.items():
        df.Label.replace(old_label, new_label, inplace=True)


def get_dataframe(first_time=False):
    
    if first_time:
        DIR_PATH = "DATA"

        FILE_NAMES = ["Monday-WorkingHours.pcap_ISCX.csv",
                        "Tuesday-WorkingHours.pcap_ISCX.csv",
                        "Wednesday-workingHours.pcap_ISCX.csv",
                        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
        
        file_name = os.path.join(DIR_PATH, "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

        _to_utf8(file_name)

        # Read dataset
        df = pd.read_csv(file_name, skipinitialspace=True)

        # Show number of NaN rows
        print("Removing {} rows that contains only NaN values...".format(df[df.isna().all(axis=1)].shape[0]))

        # Remove NaN rows
        df = df[~ df.isna().all(axis=1)]

        # Renaming labels
        _renaming_class_label(df)

        # Save to csv
        df.to_csv(file_name, index=False)

        df = [pd.read_csv(os.path.join(DIR_PATH, f), skipinitialspace=True) for f in FILE_NAMES]
        df = pd.concat(df, ignore_index=True)

        df.Label.value_counts()

        df.to_csv(os.path.join(DIR_PATH, "TrafficLabelling.csv"), index=False)
    else:
        df = pd.read_csv("DATA/TrafficLabelling.csv")

        # print label distribution
        print(df.Label.value_counts())

    return 0
