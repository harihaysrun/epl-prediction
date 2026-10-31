import pandas as pd
import sqlite3
from match_predictor_catboost_2627 import train_model
from match_predictor_catboost_2627 import predict_match

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

conn = sqlite3.connect("../epl_predictions.db")
df_prediction = pd.read_sql_query("SELECT * FROM predictions", conn)
current_matchweek = 7

matches = list(zip(df_prediction["team"], df_prediction["opponent"]))

df, teams, calibrated_model, latest, elo = train_model("all", current_matchweek)

for match in matches:
    home_team = match[0]
    away_team = match[1]
    print(f"Home: {home_team} | Away: {away_team}")
    win, draw, lose = predict_match(home_team, away_team, calibrated_model, df, latest, elo)
    print(f"{home_team} win: {win}%, \nDraw: {draw}%, \n{away_team} win: {lose}%")

    if win > draw and win > lose:
        prediction = "W"
    elif draw > win and draw > lose:
        prediction = "D"
    else:
        prediction = "L"

    print(f"home_team prediction: {prediction}")

    print("---")
    conn.execute("""
                 UPDATE predictions
                 SET prediction = ?
                 WHERE matchweek = ? AND team = ?
    """,(prediction, current_matchweek, home_team))
    conn.commit()

df_check = pd.read_sql_query("SELECT result, prediction FROM predictions WHERE matchweek = ?", conn, params=(current_matchweek,))
accuracy = (df_check["result"] == df_check["prediction"]).mean() * 100
print(f"Prediction accuracy: {accuracy:.2f}%")

df_correct = pd.read_sql_query(
    "SELECT team, result, prediction, opponent FROM predictions WHERE result = prediction AND matchweek = ?",
    conn,
    params=(current_matchweek,)
)
print(len(df_correct))
print(df_correct)

conn.execute("""
    CREATE TABLE IF NOT EXISTS weekly_accuracy (
        matchweek INTEGER,
        accuracy INTEGER,
        correct_count INTEGER,
        wrong_count INTEGER
    )
""")
conn.commit()

df_weekly_accuracy = pd.DataFrame([{
    "matchweek": current_matchweek,
    "correct_count": len(df_correct),
    "wrong_count": 20 - len(df_correct),
    "accuracy": accuracy
}])

df_weekly_accuracy.to_sql("weekly_accuracy", conn, if_exists="append", index=False)