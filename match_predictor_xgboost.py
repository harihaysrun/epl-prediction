import pandas as pd
from pprint import pprint
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

data = pd.read_excel("epl_23-26.xlsx")
df = data.sort_values("date")



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


# home.to_excel("test_home.xlsx", index=False)
# away.to_excel("test_away.xlsx", index=False)
# match_df.to_excel("test_match_df.xlsx", index=False)
# print(teams)

match_df = match_df.sort_values("date_x")
# print(match_df)
# print(match_df[match_df["team_x"] == "Man U"])

# create elo ratings
elo = {}
def get_elo(team):
    return elo.get(team, 1500)

home_elos = []
away_elos = []

K = 20

for i, row in match_df.iterrows():

    home = row["team_x"]
    away = row["opponent_x"]

    home_elo = get_elo(home)
    away_elo = get_elo(away)

    home_elos.append(home_elo)
    away_elos.append(away_elo)

    # expected probability
    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 700))

    # actual result
    if row["result"] == 0: # win
        actual = 1
    elif row["result"] == 1: # draw
        actual = 0.5
    else: # lose
        actual = 0

    elo[home] = home_elo + K * (actual - expected_home)
    elo[away] = away_elo + K * ((1 - actual) - (1 - expected_home))

match_df["home_elo"] = home_elos
match_df["away_elo"] = away_elos
match_df["elo_diff"] = match_df["home_elo"] - match_df["away_elo"]

# pprint(elo)

# pprint(match_df)

features = ["home_gls_avg5", "home_sh_avg5", "home_sot_avg5", "home_sot%_avg5", "home_g/sh_avg5", "home_g/sot_avg5", "home_pk_avg5", "home_pkatt_avg5","away_gls_avg5", "away_sh_avg5", "away_sot_avg5", "away_sot%_avg5", "away_g/sh_avg5", "away_g/sot_avg5", "away_pk_avg5", "away_pkatt_avg5", "home_elo", "away_elo", "elo_diff"]

date = "2025-08-15" # start of 25/26 season
train = match_df[match_df["date_x"] < date]
test = match_df[match_df["date_x"] >= date]

X_train = train[features]
y_train = train["result"]

X_test = test[features]
y_test = test["result"]

# train model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print("Log Loss:", log_loss(y_test, y_proba))

calibrated_model = CalibratedClassifierCV(model, method="isotonic")
calibrated_model.fit(X_train, y_train)

# y_proba_calibrated = calibrated_model.predict_proba(X_test)
# print("Calibrated Log Loss:", log_loss(y_test, y_proba_calibrated))

# evaluate
# predictions = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, predictions))
# print(confusion_matrix(y_test, predictions))

latest = df.sort_values("date").groupby("team").tail(1).set_index("team")
# print(latest)

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

        "home_elo": elo.get(home_team),
        "away_elo": elo.get(away_team),
        "elo_diff": elo.get(home_team) - elo.get(away_team)
    }])

    probs = model.predict_proba(row)[0]

    # print(row)
    print(f'''{home_team} win: {round(probs[0] * 100, 2)}%,
Draw: {round(probs[1] * 100, 2)}%,
{away_team} win: {round(probs[2] * 100, 2)}%'''
    )

# predicts probability of W/D/L using home/away stats
while True:
    print("Choose a team:")
    print(", ".join(teams))
    home_team = input("Home team: ")
    if home_team in teams:
        away_team = input("Away team: ")
        if away_team in teams:
            predict_match(home_team, away_team, calibrated_model)

            cont_loop = input("Continue? y/n: ")

            if cont_loop == "n":
                break

        else:
            print(f"{away_team} is not in the list")
    else:
        print(f"{home_team} is not in the list")