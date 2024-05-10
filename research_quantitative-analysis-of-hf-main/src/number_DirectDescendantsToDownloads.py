import pickle
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from progress.bar import Bar
from sqlalchemy import Connection, Engine, create_engine


def createDBConnection() -> Connection:
    dbPath: Path = Path("../data/peatmoss.db")
    engine: Engine = create_engine(url=f"sqlite:///{dbPath.__str__()}")
    return engine.connect()


def getDataFrames(
    conn: Connection,
) -> Tuple[DataFrame, DataFrame]:
    modelSQLQuery: str = "SELECT id, downloads FROM model;"

    models: DataFrame = (
        pandas.read_sql_query(sql=modelSQLQuery, con=conn)
        .dropna(how="any", ignore_index=True)
        .rename(columns={"id": "model_id"})
    )
    model2baseModelDF: DataFrame = pandas.read_sql_table(
        table_name="model_to_base_model",
        con=conn,
    )

    return (models, model2baseModelDF)


def computeMetric(m2: DataFrame, models: DataFrame):
    data: dict[str, List[str | int | float]] = {
        "base_model": [],
        "count": [],
        "downloads": [],
    }

    # Sanity checks the uniqueness of model-library pairings
    dfgb: DataFrameGroupBy = m2.groupby(by="base_model_name")

    with Bar("Computing metric...", max=dfgb.__len__()) as bar:
        bm: str
        df: DataFrame
        for bm, df in dfgb:
            df = df.merge(right=models, on="model_id")
            data["base_model"].append(bm)
            data["count"].append(df.shape[0])
            data["downloads"].append(int(df["downloads"].sum()))
            bar.next()

    return DataFrame(data=data).sort_values(
        by="count",
        ascending=False,
        ignore_index=True,
    )


def main() -> None:
    conn: Connection = createDBConnection()

    # Get DataFrames from SQLite database
    modelDF: DataFrame
    model2baseModelDF: DataFrame
    modelDF, model2baseModelDF = getDataFrames(conn=conn)

    # metric: DataFrame = computeMetric(m2=model2baseModelDF, models=modelDF)

    # Uncomment to save time if running multiple times
    # Write data to pickle file
    # with open("pickle/data.pickle", "wb") as pf:
    #     pickle.dump(obj=metric, file=pf)
    #     pf.close()

    # Read data from pickle file
    with open("pickle/data.pickle", "rb") as pf:
        metric: DataFrame = pickle.load(file=pf)
        pf.close()

    df: DataFrame = metric[["downloads", "count"]]

    df.to_csv(
        path_or_buf="data/number_of_direct_descendant_models_per_download.csv",
        index=False,
    )

    plt.rc("ytick", labelsize="medium")
    plt.rc("xtick", labelsize="small")

    ax: Axes = df.plot(
        kind="scatter",
        x="downloads",
        y="count",
        title="Number of Direct Descendant Models per Download",
        legend=False,
        # ylim=(0, 800),
        xlabel="Downloads",
        ylabel="Number of Direct Descendant Models",
        logx=True,
        logy=True,
    )

    fig: Figure = ax.figure

    fig.tight_layout()
    fig.savefig(fname="figs/number_of_direct_descendant_models_per_download-3.pdf")


if __name__ == "__main__":
    main()
