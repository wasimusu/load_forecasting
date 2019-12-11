import pandas as pd
import os
from dateutil import parser


def aggregate(fname, output_fname):
    if not os.path.exists(fname):
        raise ValueError("File {} does not exist".format(fname))

    df = pd.DataFrame(pd.read_csv(fname))
    columns = list(df.columns)

    aggregated_df = []
    today = parser.parse(df.loc[0][0]).date()

    energy = 0
    for i in df.index:
        new_date = parser.parse(df.loc[i][0]).date()
        new_energy = float(df.loc[i][1])
        if new_date == today:
            energy += new_energy
        else:
            aggregated_df.append([today, energy])
            energy = new_energy
            today = new_date
        if i % 1000 == 0:
            print("Processed {}/{}".format(i, len(df)))

    aggregated_df = pd.DataFrame(data=aggregated_df, columns=columns)
    aggregated_df.to_csv(path_or_buf=output_fname)


if __name__ == '__main__':
    fname = "data/household_data.csv"
    output_fname = "data/household_data_daily.csv"
    aggregate(fname, output_fname)
