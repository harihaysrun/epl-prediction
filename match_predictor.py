import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

data = pd.read_excel("epl_23-26.xlsx")
df = data.sort_values(["team", "date"])

# create rolling averages for past 5 matches
cols = ["gls", "sh", "sot", "sot%", "g/sh", "g/sot", "pk", "pkatt"]

for col in cols:
    df[col + "_avg5"] = (
        df.groupby("team")[col]
        .rolling(5)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )


# standardise team names for team & opponent columns
teams = sorted(df["team"].unique())
opponents = sorted(df["opponent"].unique())
# print(teams)
# print(opponents)

map_values = {}
for index, opp in enumerate(opponents):
    map_values[opp] = teams[index]

df["opponent"] = df["opponent"].replace(map_values)


# create unique match id so home/away can be easily merged
df["team1"] = df[["team", "opponent"]].min(axis=1)
df["team2"] = df[["team", "opponent"]].max(axis=1)
df["match_id"] = df["date"].astype(str) + "_" + df["team1"] + "_" + df["team2"]
df = df.drop(columns=["team1","team2"])
# pprint(df.shape)

# split df into home and away
home = df[df["venue"] == "Home"].copy()
away = df[df["venue"] == "Away"].copy()

home = home.rename(columns={
    "gls_avg5": "home_gls_avg5",
    "sh_avg5": "home_sh_avg5",
    "sot_avg5": "home_sot_avg5",
    "sot%_avg5": "home_sot%_avg5",
    "g/sh_avg5": "home_g/sh_avg5",
    "g/sot_avg5": "home_g/sot_avg5",
    "pk_avg5": "home_pk_avg5",
    "pkatt_avg5": "home_pkatt_avg5"
})

away = away.rename(columns={
    "gls_avg5": "away_gls_avg5",
    "sh_avg5": "away_sh_avg5",
    "sot_avg5": "away_sot_avg5",
    "sot%_avg5": "away_sot%_avg5",
    "g/sh_avg5": "away_g/sh_avg5",
    "g/sot_avg5": "away_g/sot_avg5",
    "pk_avg5": "away_pk_avg5",
    "pkatt_avg5": "away_pkatt_avg5"
})

# merge into 1 df based on match_id
match_df = home.merge(away, on=["match_id"])

# print(match_df.head())
# print(match_df.shape)
# print(match_df.isnull().sum())

def get_result(row):
    if row["gf_x"] > row["ga_x"]:
        return 0 # home win
    elif row["gf_x"] == row["ga_x"]:
        return 1 # draw
    else:
        return 2 # away win

match_df["result"] = match_df.apply(get_result, axis=1)

features = ["home_gls_avg5", "home_sh_avg5", "home_sot_avg5", "home_sot%_avg5", "home_g/sh_avg5", "home_g/sot_avg5", "home_pk_avg5", "home_pkatt_avg5","away_gls_avg5", "away_sh_avg5", "away_sot_avg5", "away_sot%_avg5", "away_g/sh_avg5", "away_g/sot_avg5", "away_pk_avg5", "away_pkatt_avg5"]

train = match_df[match_df["date_x"] < "2026-02-01"]
test = match_df[match_df["date_x"] >= "2026-02-01"]

X_train = train[features]
y_train = train["result"]

X_test = test[features]
y_test = test["result"]

# train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# evaluate
# predictions = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, predictions))
# print(confusion_matrix(y_test, predictions))

latest = df.sort_values("date").groupby("team").tail(1).set_index("team")

def predict_match(home_team, away_team, model):
    row = pd.DataFrame([{
        "home_gls_avg5": latest.loc[home_team, "gls_avg5"],
        "home_sh_avg5": latest.loc[home_team, "sh_avg5"],
        "home_sot_avg5": latest.loc[home_team, "sot_avg5"],
        "home_sot%_avg5": latest.loc[home_team, "sot%_avg5"],
        "home_g/sh_avg5": latest.loc[home_team, "g/sh_avg5"],
        "home_g/sot_avg5": latest.loc[home_team, "g/sot_avg5"],
        "home_pk_avg5": latest.loc[home_team, "pk_avg5"],
        "home_pkatt_avg5": latest.loc[home_team, "pkatt_avg5"],

        "away_gls_avg5": latest.loc[away_team, "gls_avg5"],
        "away_sh_avg5": latest.loc[away_team, "sh_avg5"],
        "away_sot_avg5": latest.loc[away_team, "sot_avg5"],
        "away_sot%_avg5": latest.loc[away_team, "sot%_avg5"],
        "away_g/sh_avg5": latest.loc[away_team, "g/sh_avg5"],
        "away_g/sot_avg5": latest.loc[away_team, "g/sot_avg5"],
        "away_pk_avg5": latest.loc[away_team, "pk_avg5"],
        "away_pkatt_avg5": latest.loc[away_team, "pkatt_avg5"],
    }])

    # print(row)
    probs = model.predict_proba(row)[0]

    print(f'''{home_team} win: {round(probs[0] * 100, 2)},
draw: {round(probs[1] * 100, 2)},
{away_team} win: {round(probs[2] * 100, 2)}'''
    )

# predicts probability of W/D/L using home/away stats
while True:
    print("Choose a team:")
    print(", ".join(sorted(teams)))
    home_team = input("Home team: ")
    away_team = input("Away team: ")

    predict_match(home_team, away_team, model)

    cont_loop = input("Continue? y/n: ")

    if cont_loop == "n":
        break