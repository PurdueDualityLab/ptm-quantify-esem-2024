from pathlib import Path
from textwrap import wrap
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import arange, logical_not
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from sqlalchemy import Connection, Engine, create_engine


def createDBConnection() -> Connection:
    dbPath: Path = Path("../data/peatmoss.db")
    engine: Engine = create_engine(url=f"sqlite:///{dbPath.__str__()}")
    return engine.connect()


def getDataFrames(
    conn: Connection,
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame,]:
    modelSQLQuery: str = "SELECT id, context_id FROM model;"

    modelDF: DataFrame = pandas.read_sql_query(sql=modelSQLQuery, con=conn)
    libraryDF: DataFrame = pandas.read_sql_table(table_name="library", con=conn)
    model2libraryDF: DataFrame = pandas.read_sql_table(
        table_name="model_to_library",
        con=conn,
    )
    model2baseModelDF: DataFrame = pandas.read_sql_table(
        table_name="model_to_base_model",
        con=conn,
    )

    return (modelDF, libraryDF, model2libraryDF, model2baseModelDF)


def identifyIfMetricIsComputable(model2library: DataFrame) -> bool:
    checkDF: DataFrame = model2library[model2library["model_id"].duplicated(keep=False)]
    if checkDF.shape[0] > 0:
        return True
    else:
        return False


def setLibraryValue(mlm: DataFrame, libraries: DataFrame) -> DataFrame:
    libraries.rename(columns={"id": "library_id"}, inplace=True)
    df: DataFrame = mlm.merge(right=libraries, on="library_id")
    df.drop(columns=["library_id"], inplace=True)
    df.rename(columns={"name": "library"}, inplace=True)
    return df


def setModelValue(mlm: DataFrame, models: DataFrame) -> DataFrame:
    models.rename(columns={"id": "model_id"}, inplace=True)
    df: DataFrame = mlm.merge(right=models, on="model_id")
    df.rename(columns={"context_id": "model"}, inplace=True)
    df.sort_values(by="model_id", inplace=True)
    return df


def dropBaseModels(mlm: DataFrame, baseModels: DataFrame) -> DataFrame:
    baseModelIDs: List[int] = baseModels["model_id"].to_list()
    df: DataFrame = mlm[mlm["model_id"].isin(values=baseModelIDs)]
    return df


def computeMetric(mlm: DataFrame):
    data: dict[str, List[str | int | float]] = {
        "library": [],
        "count": [],
        "prop": [],
    }

    # Sanity checks the uniqueness of model-library pairings
    modelCount: int = len(mlm["model"].unique())
    dfgb: DataFrameGroupBy = mlm.groupby(by="library")
    lib: str
    df: DataFrame
    for lib, df in dfgb:
        if lib == "sentence-transformers":
            data["library"].append("sentence-tx")
        else:
            data["library"].append(lib)
        data["count"].append(len(df["model"].unique()))
        data["prop"].append(len(df["model"].unique()) / modelCount)
    return DataFrame(data=data).sort_values(
        by="prop",
        ascending=False,
        ignore_index=True,
    )


def main() -> None:
    conn: Connection = createDBConnection()

    modelDF: DataFrame
    libraryDF: DataFrame
    model2libraryDF: DataFrame  # 166,705 models
    model2baseModelDF: DataFrame

    # Get DataFrames from SQLite database
    modelDF, libraryDF, model2libraryDF, model2baseModelDF = getDataFrames(conn=conn)

    # Make sure that there is at least one model that leveraged more than one
    # library
    assert identifyIfMetricIsComputable(model2library=model2libraryDF)

    # Set values inplace of library ids
    mlmDF: DataFrame = setLibraryValue(
        mlm=model2libraryDF,
        libraries=libraryDF,
    )

    # Set values inplace of model ids
    mlmDF: DataFrame = setModelValue(
        mlm=mlmDF,
        models=modelDF,
    )

    # Drop base models (only keep descendents)
    mlmDF: DataFrame = dropBaseModels(mlm=mlmDF, baseModels=model2baseModelDF)

    # Drop models that use only 1 library
    mlmDF: DataFrame = mlmDF[mlmDF["model_id"].duplicated(keep=False)]

    # Compute metric with normalized values
    metric: DataFrame = computeMetric(mlm=mlmDF)

    plt.rc("ytick", labelsize="xx-large")
    plt.rc("xtick", labelsize="xx-large")
    plt.rc("figure", titlesize="xx-large")
    plt.rc("axes", labelsize="xx-large")
    plt.rc("axes", titlesize="xx-large")

    df: DataFrame = DataFrame(data=metric[["library", "prop"]][0:10])

    df.set_index(keys=["library"], inplace=True)
    print(df.index)

    ax: Axes = df.plot(
        kind="bar",
        title="Proportion of Direct Descendant Models per Library",
        legend=False,
        ylim=(0, 1),
        xlabel="Library",
        ylabel="\n".join(
            wrap(
                text="Proportion of Direct Descendant Models",
                width=20,
            )
        ),
        width=0.75,
        align="center",
    )
    ax.set_xticklabels(df.index, rotation=45, ha="right", rotation_mode="anchor")

    idx: int
    for idx in range(10):
        ax.text(
            x=idx,
            y=df["prop"].iloc[idx] + 0.005,
            s=metric["count"].iloc[idx],
            fontsize="x-large",
            ha="center",
        )

    fig: Figure = ax.figure

    fig.set_size_inches(w=8, h=6)

    fig.tight_layout()
    fig.savefig(fname="figs/proportion_of_direct_descendant_models_per_library-15.pdf")


if __name__ == "__main__":
    main()
