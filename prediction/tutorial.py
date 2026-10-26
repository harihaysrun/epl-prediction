import pandas as pd
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
# pd.set_option("display.expand_frame_repr", False)

matches = pd.read_csv("matches.csv", index_col=0)
# matches = pd.read_excel("epl_23-26.xlsx", index_col=0)
# pprint(matches)
# print(matches.columns)

# print(matches["team"].value_counts())
# print(matches["round"].value_counts())
matches["date"] = pd.to_datetime(matches["date"])
# pprint(matches.dtypes)

# convert home/away col into numeric col
matches["venue_code"] = matches["venue"].astype("category").cat.codes

# convert opponent into integer
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

# to see if teams perform better at certain timings
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

# get no. of day in the week
matches["day_code"] = matches["date"].dt.dayofweek

# target: if the team won or not
matches["target"] = (matches["result"] == "W").astype("int") # will turn true into 1 and false into 0
# print(matches)

# series of decision trees, each has slightly different params
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# train algorithm based on data from the past
train = matches[matches["date"] < "2022-01-01"]
test = matches[matches["date"] > "2022-01-01"]
predictors = ["venue_code", "opp_code", "hour", "day_code"]

rf.fit(train[predictors], train["target"])
predictions = rf.predict(test[predictors])

# determine accuracy of the model
accuracy = accuracy_score(test["target"], predictions)
# print(accuracy)

combined = pd.DataFrame(dict(actual=test["target"], prediction=predictions), index=test.index)
# pd.crosstab(index=combined["actual"],columns=combined["prediction"])

# precision_score: what percentage of the time did the team actually win
prec = precision_score(test["target"], predictions)
# print(prec)

# split df by teams
grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester City")
# pprint(group.columns)

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date") # look at the last 3 matches the team played
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
# pprint(new_cols)

# example for Manchester City
man_city = rolling_averages(group, cols, new_cols)
# pprint(man_city)

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x,cols,new_cols))
# matches_rolling = matches_rolling.droplevel("team") # drop extra index level
matches_rolling = matches_rolling.reset_index(level="team")
matches_rolling.index = range(matches_rolling.shape[0]) # assign new index for each row
# pprint(matches_rolling.columns)

# retrain model
def make_predictions(data, predictors):
    train = data[data["date"] < "2022-01-01"]
    test = data[data["date"] > "2022-01-01"]
    rf.fit(train[predictors], train["target"])
    predictions = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=predictions), index=test.index)
    precision = precision_score(test["target"], predictions)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
# print(combined)

# home and away matches

# team name listed in team and opponent cols may be diff
# e.g Wolverhampton Wanderers | Wolves
class MissingDict(dict):
    __missing__ = lambda self,key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}

mapping = MissingDict(**map_values)
# pprint(mapping["Wolverhampton Wanderers"])

combined["team"] = combined["team"].map(mapping)
# pprint(combined)

# merge df with itself
# there may be duplicated rows e.g
# row no.  |      date      |    team     |   opponent
#   55     |   2022-01-23   |   Arsenal   |    Burnley
#  1000    |   2022-01-23   |   Burnley   |    Arsenal

merged = combined.merge(combined, left_on=["date", "team"], right_on=["date", "opponent"])
pprint(len(merged))

# find rows where 1 team is predicted to win and the other is predicted to lose as they're more accurate
acc_pred = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()
pprint(acc_pred / sum(acc_pred))

accurate_predictions = merged.loc[(merged["actual_x"] == 1) & (merged["predicted_x"] == 1) & (merged["actual_y"] == 0) & (merged["predicted_y"] == 0)]
pprint(accurate_predictions)

# TODO: choose a Matchweek as the "next match" to test if prediction is accurate based on past 3 matches