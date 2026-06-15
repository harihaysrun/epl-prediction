import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_excel("epl_23-26.xlsx")
team_names = data["team"].unique()

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

# pprint(df.head(20))

mapping = {"W": 0,"D": 1,"L": 2}
df["result_cat"] = df["result"].map(mapping)

features = [f"{c}_avg5" for c in cols]
X = df[features]
y = df["result_cat"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy_score(y_test, predictions)
probs = model.predict_proba(X_test)

test = confusion_matrix(y_test, predictions)
# pprint(test)

# this prediction [W/D/L] is only for one team
while True:
    match_date = input("Match date in YYYY-MM-DD: ")
    print("Choose a team:")
    pprint(", ".join(sorted(team_names)))
    team = input("Enter team name: ")

    # get the most recent row for this team, before selected date
    team_stats = df[ (df["team"] == team) & (df["date"] < match_date) ].sort_values("date").iloc[-1]

    X_match = pd.DataFrame([{
        "gls_avg5": team_stats["gls_avg5"],
        "sh_avg5": team_stats["sh_avg5"],
        "sot_avg5": team_stats["sot_avg5"],
        "sot%_avg5": team_stats["sot%_avg5"],
        "g/sh_avg5": team_stats["g/sh_avg5"],
        "g/sot_avg5": team_stats["g/sot_avg5"],
        "pk_avg5": team_stats["pk_avg5"],
        "pkatt_avg5": team_stats["pkatt_avg5"],
    }])

    wdl_prediction = model.predict_proba(X_match)[0]
    print(f"WIN : {wdl_prediction[0]}")
    print(f"DRAW: {wdl_prediction[1]}")
    print(f"LOSE: {wdl_prediction[2]}")

    # print actual result from original df
    pprint(df[ (df["team"] == team) & (df["date"] == match_date) ][["team", "date", "result"]])

    cont_loop = input("Continue? y/n: ")

    if cont_loop == "n":
        break