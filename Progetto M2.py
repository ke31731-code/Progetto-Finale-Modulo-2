"""
MEGASHOP - Data Engineering Pipeline (ESAME PRATICO)
=======================================================

T
ESECUZIONE:
    python megashop_pipeline.py            -> esegue Esercizi 1, 2, 3 in sequenza
    python megashop_pipeline.py 1          -> solo Esercizio 1 (Pandas vs Dask)
    python megashop_pipeline.py 2          -> solo Esercizio 2 (ETL PySpark)
    python megashop_pipeline.py 3          -> solo Esercizio 3 (Visualization)
    python megashop_pipeline.py 4          -> Esercizio 4 Bonus (Streaming, bloccante)

"""

import os
import sys
import glob

# ======================================================================
# VARIABILI D'AMBIENTE
# ======================================================================
os.environ["JAVA_HOME"] = r"C:\Users\ke317\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
os.environ["HADOOP_HOME"] = r"C:\hadoop"

os.environ["PATH"] = (
    os.environ["JAVA_HOME"] + r"\bin;"
    + r"C:\hadoop\bin;"
    + os.environ["PATH"]
)

os.environ["PYSPARK_PYTHON"] = r"C:\Users\ke317\AppData\Local\Programs\Python\Python310\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\ke317\AppData\Local\Programs\Python\Python310\python.exe"

import pandas as pd
import dask.dataframe as dd

import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as spark_sum, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType


# ======================================================================
# CONFIGURAZIONE - path coerenti con il generator.py fornito
# ======================================================================
BASE_DIR = "./data_local"
PARQUET_DIR = os.path.join(BASE_DIR, "parquet")
JSON_DIR = os.path.join(BASE_DIR, "json")

PRODOTTI_PARQUET = os.path.join(PARQUET_DIR, "products.parquet")
CUSTOMERS_PARQUET = os.path.join(PARQUET_DIR, "customers.parquet")  
REGIONI_PARQUET = os.path.join(PARQUET_DIR, "regions.parquet")
TRANSAZIONI_PARQUET_PATTERN = os.path.join(PARQUET_DIR, "transactions_batch_*.parquet")
TRANSAZIONI_JSON_PATTERN = os.path.join(JSON_DIR, "transactions_part_*.jsonl")

PROCESSED_SALES_DIR = os.path.join(BASE_DIR, "processed_sales")
OUTPUT_IMAGE = "fatturato_per_categoria.png"


GROUP_COLUMN_EX1 = "region_id"


# ======================================================================
# ESERCIZIO 1 - Ingestion e Limiti di Memoria (Pandas vs Dask)
# ======================================================================
def esercizio1_pandas():
    print("=" * 70)
    print("ESERCIZIO 1.1 - PANDAS: lettura file per file (ciclo for)")
    print("=" * 70)

    json_files = sorted(glob.glob(TRANSAZIONI_JSON_PATTERN))
    if not json_files:
        print(f"Nessun file trovato in: {TRANSAZIONI_JSON_PATTERN}")
        return None

    print(f"Trovati {len(json_files)} file. Inizio elaborazione...\n")

    totale_generale = 0.0
    for file_path in json_files:
        df = pd.read_json(file_path, lines=True)
        totale_file = df["amount"].sum()
        print(f"  {os.path.basename(file_path):35s} -> totale = {totale_file:,.2f}")
        totale_generale += totale_file

    print(f"\nTOTALE GENERALE (Pandas): {totale_generale:,.2f}")
    return totale_generale


def esercizio1_dask():
    print("\n" + "=" * 70)
    print("ESERCIZIO 1.2 - DASK: lettura con wildcard")
    print("=" * 70)

    ddf = dd.read_json(TRANSAZIONI_JSON_PATTERN, lines=True)
    print("Numero di partizioni Dask:", ddf.npartitions)

    risultato = (
        ddf.groupby(GROUP_COLUMN_EX1)["amount"]
        .mean()
        .compute()  # qui scatta il calcolo effettivo (Dask è lazy)
    )

    print(f"\nMedia importo per '{GROUP_COLUMN_EX1}':")
    print(risultato)
    return risultato


def run_esercizio1():
    esercizio1_pandas()
    esercizio1_dask()


# ======================================================================
# ESERCIZIO 2 - Pipeline ETL con PySpark
# ======================================================================
def run_esercizio2(spark=None):
    print("\n" + "=" * 70)
    print("ESERCIZIO 2 - ETL con PySpark")
    print("=" * 70)

    own_session = spark is None
    if own_session:
        spark = (
            SparkSession.builder
            .appName("MegaShop-ETL")
            .master("local[*]")
            # Di default Spark crea 200 partizioni per ogni shuffle (join,
            # groupBy...). Su un PC normale (pochi core) questo genera un
            # overhead enorme di task piccolissimi e, scrivendo partizionato
            # per "year", anche centinaia di mini-file inutili. Lo riduciamo
            # a un valore sensato per esecuzione locale.
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

    # ---------------- EXTRACT ----------------
    batch_files = sorted(glob.glob(TRANSAZIONI_PARQUET_PATTERN))
    if not batch_files:
        raise FileNotFoundError(f"Nessun file trovato: {TRANSAZIONI_PARQUET_PATTERN}")

    pdf_transazioni = pd.concat(
        (pd.read_parquet(f) for f in batch_files),
        ignore_index=True,
    )
    
    pdf_transazioni = pdf_transazioni.drop(columns=["ts"], errors="ignore")

    transazioni = spark.createDataFrame(pdf_transazioni)
    prodotti = spark.read.parquet(PRODOTTI_PARQUET)
    regioni = spark.read.parquet(REGIONI_PARQUET)

    print("\nSchema transazioni:")
    transazioni.printSchema()

    # ---------------- TRANSFORM ----------------
    df = transazioni.join(prodotti, on="product_id", how="left")
    df = df.join(regioni, on="region_id", how="left")

    final_df = df.select(
        "transaction_id",
        "region_name",
        "category",
        "amount",
        "year",
    )

   
    final_df = final_df.cache()

    print("\nAnteprima DataFrame finale:")
    final_df.show(10, truncate=False)
    print(f"Righe totali: {final_df.count()}")

    # ---------------- LOAD ----------------
    print(f"\nSalvo in '{PROCESSED_SALES_DIR}' partizionato per year...")
    (
        final_df
        .coalesce(4)  # evita di scrivere tanti mini-file per partizione
        .write
        .mode("overwrite")
        .partitionBy("year")
        .parquet(PROCESSED_SALES_DIR)
    )
    print("Fatto.")

    if own_session:
        spark.stop()
        return None

    return final_df


# ======================================================================
# ESERCIZIO 3 - Data Visualization (Reporting)
# ======================================================================
def run_esercizio3(spark=None, final_df=None):
    print("\n" + "=" * 70)
    print("ESERCIZIO 3 - Data Visualization")
    print("=" * 70)

    own_session = spark is None
    if own_session:
        spark = (
            SparkSession.builder
            .appName("MegaShop-Reporting")
            .master("local[*]")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

    if final_df is None:
        final_df = spark.read.parquet(PROCESSED_SALES_DIR)

    fatturato_per_categoria = (
        final_df.groupBy("category")
        .agg(spark_sum("amount").alias("fatturato_totale"))
        .orderBy("fatturato_totale", ascending=False)
    )

    print("Fatturato totale per categoria:")
    fatturato_per_categoria.show()

    # Risultato aggregato in Pandas.
    pdf = fatturato_per_categoria.toPandas()

    if own_session:
        spark.stop()

    # ---------------- PLOT ----------------
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pdf, x="category", y="fatturato_totale", color="steelblue")
    plt.title("Fatturato Totale per Categoria")
    plt.xlabel("Categoria")
    plt.ylabel("Fatturato Totale (€)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"\nGrafico salvato come '{OUTPUT_IMAGE}'")
    plt.show() 


# ======================================================================
# ESERCIZIO 4 (BONUS) - Real-Time Streaming
# ======================================================================
# Schema coerente con i campi presenti in transactions_part_*.jsonl
STREAMING_SCHEMA = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("region_id", IntegerType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("amount", DoubleType(), True),
    StructField("ts", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("month", IntegerType(), True),
])


def run_esercizio4():
    print("\n" + "=" * 70)
    print("ESERCIZIO 4 (BONUS) - Real-Time Streaming")
    print("=" * 70)

    spark = (
        SparkSession.builder
        .appName("MegaShop-Streaming")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    streaming_df = (
        spark.readStream
        .schema(STREAMING_SCHEMA)
        .json(JSON_DIR)
    )

    conteggio_per_regione = (
        streaming_df.groupBy("region_id")
        .agg(count("*").alias("totale_transazioni"))
    )

    query = (
        conteggio_per_regione.writeStream
        .outputMode("complete")  # "complete" perché è un'aggregazione
        .format("console")
        .start()
    )

    print(f"In ascolto su '{JSON_DIR}' ... (Ctrl+C per terminare)")
    query.awaitTermination()


# ======================================================================
# MAIN
# ======================================================================
def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "1":
        run_esercizio1()

    elif arg == "2":
        run_esercizio2()

    elif arg == "3":
        run_esercizio3()

    elif arg == "4":
        run_esercizio4()

    elif arg is None:
        # Sequenza completa 1 -> 2 -> 3 
        run_esercizio1()

        spark = (
            SparkSession.builder
            .appName("MegaShop-Pipeline")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        final_df = run_esercizio2(spark=spark)
        run_esercizio3(spark=spark, final_df=final_df)

        spark.stop()

        print("\nEsercizi 1-3 completati.")
        

    else:
        print("Argomento non valido. Usa: 1, 2, 3, 4 oppure nessun argomento.")


if __name__ == "__main__":
    main()