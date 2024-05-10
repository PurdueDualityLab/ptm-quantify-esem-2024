from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from sqlalchemy import Connection, Engine, create_engine


def createDBConnection() -> Connection:
    dbPath: Path = Path("../data/peatmoss.db")
    engine: Engine = create_engine(url=f"sqlite:///{dbPath.__str__()}")
    return engine.connect()


def getDataFrames(
    conn: Connection,
) -> DataFrame:
    model2baseModelDF: DataFrame = pandas.read_sql_table(
        table_name="model_to_base_model",
        con=conn,
    )

    return model2baseModelDF


def computeMetric(m2: DataFrame):
    data: dict[str, List[str | int | float]] = {
        "base_model": [],
        "count": [],
    }

    # Sanity checks the uniqueness of model-library pairings
    dfgb: DataFrameGroupBy = m2.groupby(by="base_model_name")

    bm: str
    df: DataFrame
    for bm, df in dfgb:
        data["base_model"].append(bm)
        data["count"].append(df.shape[0])
    return DataFrame(data=data).sort_values(
        by="count",
        ascending=False,
        ignore_index=True,
    )


def main() -> None:
    conn: Connection = createDBConnection()

    # Get DataFrames from SQLite database
    model2baseModelDF: DataFrame = getDataFrames(conn=conn)

    metric: DataFrame = computeMetric(m2=model2baseModelDF)

    df: DataFrame = metric[["base_model", "count"]][0:10]
    df.set_index(keys=["base_model"], inplace=True)

    plt.rc("ytick", labelsize="medium")
    plt.rc("xtick", labelsize="small")

    ax: Axes = df.plot(
        kind="bar",
        title="Number of Direct Descendant Models per Parent Model",
        legend=False,
        ylim=(0, 800),
        xlabel="Parent Model",
        ylabel="Number of Direct Descendant Models",
        width=0.75,
        align="center",
    )

    idx: int
    for idx in range(10):
        ax.text(
            x=idx,
            y=df["count"].iloc[idx] + 5,
            s=metric["count"].iloc[idx],
            fontsize="small",
            ha="center",
        )

    fig: Figure = ax.figure

    fig.tight_layout()
    fig.savefig(fname="figs/number_of_direct_descendant_models_per_parent_model.pdf")


if __name__ == "__main__":
    main()
