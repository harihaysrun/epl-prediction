import pandas as pd
from pprint import pprint

files = ["23-24.xlsx", "24-25.xlsx", "25-26.xlsx"]

game_data = []

# loop through each file
for index, file in enumerate(files):
    all_sheets = pd.read_excel(file, sheet_name=None)

    # loop through each sheet/team and add to game_data list
    for team_name, data in all_sheets.items():
        data["Team"] = team_name
        data["Year"] = data["Date"].dt.year
        data["Matchweek"] = data["Round"].str.split().str[-1]
        data["Season"] = file.strip(".xlsx").replace("-", "/")
        game_data.append(data)

df = pd.concat(game_data, ignore_index=True)
df.columns = df.columns.str.lower()
df_filtered = df[df["comp"] == "Premier League"]
df_final = df_filtered.drop(columns=["comp", "round", "referee", "match report", "notes"])
pprint(df_final)

df_final.to_excel("epl_23-26.xlsx", index=False)