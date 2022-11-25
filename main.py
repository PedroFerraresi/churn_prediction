from datetime import datetime

import joblib
import pandas as pd


def read_file(file):
    return pd.read_csv(file, delimiter=";")


def load_pipeline(pipeline):
    return joblib.load(pipeline)


def predict_churn(dataframe, pipeline):
    dataframe["predictedValues"] = pipeline.predict(dataframe)

    return dataframe


def export_predictions(dataframe):
    date = datetime.now().date().strftime("%Y%m%d")

    dataframe.loc[:, ["RowNumber", "predictedValues"]].to_csv(
        f"./data/answer/abandonos_{date}.csv", index=False
    )

    return None


def main():
    # Load CSV File
    df = read_file(f"./data/test/abandono_teste.csv")

    # Load Pipeline
    pipeline = load_pipeline(f"./models/pipeline.pkl")

    # Predict Churns
    df = predict_churn(df, pipeline)

    # Export Answers
    export_predictions(df)


if __name__ == "__main__":
    main()
