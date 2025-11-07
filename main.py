import os
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
import pandas as pd
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
# ------------------------------------------------------------------
# 1️⃣  Cached factory – creates ONE session for the whole app
# ------------------------------------------------------------------
@st.cache_resource()
def get_spark_session() -> SparkSession:
    """
    Create (once) and return a SparkSession.
    This function is executed only once; subsequent calls return the cached object.
    """
    print("--- CRÉATION D'UNE NOUVELLE SESSION SPARK ---")  # Debug

    # You can read an env var if you ever want to connect to a cluster
    spark_master_url = os.environ.get("SPARK_MASTER_URL", "local[*]")

    return (
        SparkSession.builder
        .appName("AttritionPrediction")
        .master(spark_master_url)
        .getOrCreate()
    )


# ------------------------------------------------------------------
# 2️⃣  Retrieve the session (will use the cached one after first run)
# ------------------------------------------------------------------
spark = get_spark_session()


# ------------------------------------------------------------------
# 3️⃣  Your Streamlit UI
# ------------------------------------------------------------------
st.title("Projet de Prédiction d'Attrition")

st.write(f"Version de Spark: {spark.version}")

# Example: show a tiny dataframe
df = spark.read.csv("/app/data-set.csv",header=True, inferSchema=True,sep=',')


st.dataframe(df.toPandas())
# st.text(df.schema.simpleString())

description = df.describe()
st.dataframe( description.toPandas())
schema = df.schema

numeric_cols = [f.name for f in schema if isinstance(f.dataType ,( IntegerType, DoubleType))]
string_cols = [f.name for f in schema if isinstance(f.dataType ,StringType)]
cols_json = {
"numeric_cols" : numeric_cols ,
"string_cols" : string_cols ,
}
st.json(cols_json)
st.text(numeric_cols)
st.subheader("Statistiques (uniquement pour les colonnes numériques)")
if numeric_cols:
    st.dataframe(df.select(numeric_cols).describe().toPandas())
st.text('after analyzing the cols i see that the rowNumber ,  CustomerId  and Surname cols are not necessaire for the data analyse ')
st.text('so we drop the cols or create a new data frame just with the most important features')
new_df = df.select("CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited")
new_schema = new_df.schema

numeric_cols = [f.name for f in new_schema if isinstance(f.dataType ,( IntegerType, DoubleType))]
string_cols = [f.name for f in new_schema if isinstance(f.dataType ,StringType)]
st.dataframe(new_df.toPandas())
null_counts_exprs = [
        F.count(
            F.when(
                F.col(c).isNull() | F.isnan(c),  # La valeur est Null OU NaN
                1                                # Si oui, marque-la d'un '1'
            )
        ).alias(c)  # Nomme le résultat d'après la colonne
        for c in df.columns # Fait cela pour toutes les colonnes
    ]


null_counts_df = df.select(null_counts_exprs)
st.text("les nombres de valeur manquants pour chaque col")
# 3. Convertit le résultat en Pandas pour l'afficher dans Streamlit
st.dataframe(null_counts_df.toPandas())
quantiles = new_df.approxQuantile(numeric_cols, [0.25, 0.75], 0.001)
st.text(quantiles)
quantiles_dict = dict(zip(numeric_cols,quantiles))
st.json(quantiles_dict)
st.subheader("Colonnes groupées par Type (de new_df)")

count_expressions = []

for col in numeric_cols:
    if col in quantiles_dict and quantiles_dict[col] is not None:
        q10 = quantiles_dict[col][0]  # 10%
        q90 = quantiles_dict[col][1]  # 90%

        condition = F.col(col).between(q10, q90)

        count_expressions.append(
            F.count(F.when(condition, 1)).alias(f"{col}_INLIERS")
        )

        count_expressions.append(
            F.count(F.when(~condition, 1)).alias(f"{col}_OUTLIERS")
        )
        st.write("Démarrage du Job 2 : Comptage de tous les inliers/outliers...")
        if count_expressions:
            counts_df = new_df.agg(*count_expressions)  # .agg() est fait pour ça
            st.dataframe(counts_df.toPandas())
        else:
            st.write("Aucune colonne numérique à analyser.")
for col in numeric_cols:
    st.write(f"Histogramme pour : {col}")
    try:
        hist_data = new_df.select(col).rdd.flatMap(lambda x: x).histogram(20)
        bin_edges = hist_data[0]
        counts = hist_data[1]

        # --- ÉTAPE 2 : PRÉPARER ET AFFICHER AVEC PANDAS & STREAMLIT ---

        # Créez des étiquettes pour les bacs (ex: "18.0 - 22.5")
        bin_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}" for i in range(len(counts))]

        # Créez un DataFrame Pandas pour le graphique
        plot_df = pd.DataFrame({
            f'Bacs ({col})': bin_labels,
            'Comptage': counts
        }).set_index(f'Bacs ({col})')  # st.bar_chart() aime avoir un index

        # Affichez le graphique !
        st.bar_chart(plot_df)
    except Exception as e:
        st.error(f"Erreur lors du calcul de l'histogramme pour {col}: {e}")

st.write("Démarrage du Job 2 : Comptage de tous les inliers/outliers...")
if count_expressions:
    counts_df = new_df.agg(*count_expressions)  # .agg() est fait pour ça

    st.dataframe(counts_df.toPandas())
else:
    st.write("Aucune colonne numérique à analyser.")
try:

    schema = new_df.schema


    col_types = {
        "Colonnes Numériques": numeric_cols,
        "Colonnes Textuelles (String)": string_cols,
    }

    st.json(col_types)
    for col in numeric_cols:
        st.text(col)
        quantiles = new_df.approxQuantile(col, [0.1, 0.9], 0.001)
        condition = new_df[col].between(quantiles[0],quantiles[-1])
        number_of_outliers = new_df.filter(condition).count()
        st.text(f"number of row are in the interval are :  {number_of_outliers}")
        number_of_outliers = new_df.filter(~condition).count()
        st.text(f"number of row are not in the interval are :  {number_of_outliers}")

except Exception as e:
    st.error(f"Erreur lors du tri des types de colonnes : {e}")
st.dataframe(new_df.groupBy('Age').count())
st.dataframe(new_df.select('Geography').distinct().toPandas())
indexers = [StringIndexer(inputCol="Geography", outputCol="GeographyIndex"),
StringIndexer(inputCol="Gender", outputCol="GenderIndex")
            ]
pipeline_idx = Pipeline(stages=indexers)
df_indexed = pipeline_idx.fit(new_df).transform(new_df)
ohe = OneHotEncoder(
    inputCols=["GeographyIndex","GenderIndex"],
    outputCols=["GeographyVec","GenderVec"]
)
df_ohe = ohe.fit(df_indexed).transform(df_indexed)
st.dataframe(df_ohe.toPandas())
