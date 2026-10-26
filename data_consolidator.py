import pandas as pd
from pprint import pprint
import sqlite3

files = ["23-24.xlsx", "24-25.xlsx", "25-26.xlsx", "26-27.xlsx"]

current_matchweek = 7
game_data = []

# loop through each file
for index, file in enumerate(files):
    all_sheets = pd.read_excel(file, sheet_name=None)

    # loop through each sheet/team and add to game_data list
    for team_name, data in all_sheets.items():
        data["Team"] = team_name
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
        print(data["Date"].dtype)
        print(team_name)
        print(data["Date"])
        data["Year"] = data["Date"].dt.year
        data["Matchweek"] = data["Round"].str.split().str[-1]
        data["Season"] = file.strip(".xlsx").replace("-", "/")
        if file == "26-27.xlsx":
            # print(data["Matchweek"].astype(int))
            data = data[data["Matchweek"].str.isnumeric()]
            data = data[data["Matchweek"].astype(int) <= current_matchweek]
            game_data.append(data)
        else:
            game_data.append(data)

df = pd.concat(game_data, ignore_index=True)
df.columns = df.columns.str.lower()
df_filtered = df[df["comp"] == "Premier League"]

# get data from 23/24 to current 26/27 matchweek
df_final = df_filtered.drop(columns=["comp", "round", "referee", "match report", "notes"])
pprint(df_final)
df_final.to_excel("epl_23-27.xlsx", index=False)

# get data for 26/27 match prediction vs results table
cols = ["date","matchweek","team","result","opponent"]

current_matchweek_table = df_filtered[(df_filtered["season"] == "26/27") & (df_filtered["matchweek"].astype(int) == current_matchweek)][cols]
current_matchweek_table["prediction"] = None
print(current_matchweek_table)

# standardise team names for team & opponent columns
df_compiled = df_final[df_final["season"] == "26/27"]

teams = sorted(df_compiled["team"].unique())
opponents = sorted(current_matchweek_table["opponent"].unique())

map_values = {}
for index, opp in enumerate(opponents):
    map_values[opp] = teams[index]

current_matchweek_table["opponent"] = current_matchweek_table["opponent"].replace(map_values)
pprint(current_matchweek_table)

conn = sqlite3.connect("epl_predictions.db")
conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        date TEXT,
        matchweek INTEGER,
        team TEXT,
        result TEXT,
        opponent TEXT,
        prediction TEXT
    )
""")
conn.commit()

current_matchweek_table.to_sql("predictions", conn, if_exists="append", index=False)
