# -*- coding: utf-8 -*-
"""
Tarea 3 - Procesamiento de Datos con Apache Spark (Big Data - UNAD, 202016911)
Autor: Daniel Andrés Mercado Trillos

SCRIPT 2 DE 2: ANÁLISIS EXPLORATORIO (EDA) Y VISUALIZACIÓN (procesamiento en batch)

Problema:
    ¿Cómo ha evolucionado la demanda de servicios de salud en Colombia entre
    2009 y 2021 según el tipo de atención, el departamento y el diagnóstico,
    y qué impacto tuvo la pandemia de 2020 en esa demanda?

Entrada:
    Datos limpios en Parquet generados por 01_limpieza_rips.py (HDFS).

Salidas:
    - Resultados impresos en consola (tablas de Spark).
    - Gráficas PNG en la carpeta ./graficas
    - Tablas resumen CSV en la carpeta ./resultados

Preguntas que responde:
    P1. ¿Cuántas atenciones hubo por año y cómo varió la tendencia?
    P2. ¿Qué tipo de atención concentra la mayor demanda?
    P3. ¿Qué departamentos registran más atenciones? (énfasis en el Cesar)
    P4. ¿Cuáles son los 10 diagnósticos más frecuentes, en total y en urgencias?
    P5. ¿Cómo cambió la demanda en 2020 frente a 2019?
    Complementos: capítulos CIE-10, regiones y municipios del Cesar (con RDD).

Ejecución:
    spark-submit --driver-memory 2g 02_analisis_rips.py
    (opcional) spark-submit 02_analisis_rips.py <ruta_parquet>
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # la VM no tiene entorno gráfico: se guardan las imágenes en PNG
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RUTA_PARQUET = sys.argv[1] if len(sys.argv) > 1 else "hdfs://localhost:9000/Tarea3/rips_limpio"
DIR_GRAFICAS = "graficas"
DIR_RESULTADOS = "resultados"
DEPARTAMENTO_ENFASIS = "20"  # Código DANE del Cesar

COLOR_PRINCIPAL = "#1f6f8b"
COLOR_ENFASIS = "#e07a1f"
COLORES = ["#1f6f8b", "#e07a1f", "#3a9d5d", "#b23a48", "#7d5ba6", "#8c8c8c"]

# Formato de los ejes en millones: 1234567 -> "1.2 M"
millones = FuncFormatter(lambda x, _: f"{x / 1e6:,.1f} M")


def titulo(texto):
    """Imprime un encabezado visible en la consola para separar cada análisis."""
    print("\n" + "=" * 78)
    print(f"  {texto}")
    print("=" * 78)


def guardar_grafica(nombre):
    """Guarda la figura actual en la carpeta de gráficas y la cierra."""
    ruta = os.path.join(DIR_GRAFICAS, nombre)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  -> Gráfica guardada: {ruta}")


def guardar_tabla(pdf, nombre):
    """Guarda un resultado pequeño (pandas) como CSV para el informe."""
    ruta = os.path.join(DIR_RESULTADOS, nombre)
    pdf.to_csv(ruta, index=False, encoding="utf-8-sig")


def crear_sesion():
    spark = (
        SparkSession.builder
        .appName("RIPS - Analisis exploratorio")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def cargar(spark):
    """Lee el Parquet limpio y construye un 'cubo' agregado reutilizable."""
    titulo("CARGA DE LOS DATOS LIMPIOS (PARQUET)")
    df = spark.read.parquet(RUTA_PARQUET)
    df.printSchema()

    # Cubo agregado: reduce ~38 millones de filas a unas decenas de miles.
    # Se guarda en caché porque P1, P2, P3, P5 y los capítulos salen de aquí,
    # así Spark no vuelve a leer todo el Parquet en cada análisis.
    cubo = (
        df.groupBy("anio", "region", "cod_departamento", "departamento_nombre",
                   "tipo_atencion", "capitulo_cie10")
        .agg(F.sum("numero_atenciones").alias("atenciones"),
             F.count(F.lit(1)).alias("registros"))
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    print(f"Filas del cubo agregado en caché: {cubo.count():,}")
    return df, cubo


def estadisticas_generales(df):
    titulo("ESTADÍSTICAS DESCRIPTIVAS DE numero_atenciones")
    df.select("numero_atenciones").describe().show()
    # Percentiles aproximados (algoritmo distribuido de Spark)
    p = df.approxQuantile("numero_atenciones", [0.25, 0.5, 0.75, 0.95, 0.99], 0.01)
    print(f"  Percentiles 25/50/75/95/99: {p}")
    print("  Interpretación: la mayoría de registros agrupan pocas atenciones, pero")
    print("  unos pocos concentran volúmenes muy altos (distribución asimétrica).")


def p1_atenciones_por_anio(cubo):
    titulo("P1. ATENCIONES POR AÑO Y VARIACIÓN ANUAL")
    ventana = Window.orderBy("anio")
    por_anio = (
        cubo.groupBy("anio").agg(F.sum("atenciones").alias("atenciones"))
        .withColumn("anterior", F.lag("atenciones").over(ventana))
        .withColumn("variacion_pct",
                    F.round((F.col("atenciones") - F.col("anterior")) / F.col("anterior") * 100, 1))
        .drop("anterior")
        .orderBy("anio")
    )
    por_anio.show(20)
    pdf = por_anio.toPandas()  # solo 13 filas: seguro llevarlas al driver
    guardar_tabla(pdf, "p1_atenciones_por_anio.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pdf["anio"], pdf["atenciones"], marker="o", color=COLOR_PRINCIPAL, linewidth=2)
    ax.axvspan(2019.5, 2021.5, color=COLOR_ENFASIS, alpha=0.12, label="Pandemia")
    ax.yaxis.set_major_formatter(millones)
    ax.set_xticks(pdf["anio"])
    ax.set_title("Atenciones en salud por año (RIPS, Colombia)")
    ax.set_xlabel("Año")
    ax.set_ylabel("Atenciones (millones)")
    ax.grid(alpha=0.3)
    ax.legend()
    guardar_grafica("p1_atenciones_por_anio.png")


def p2_tipo_atencion(cubo):
    titulo("P2. DEMANDA POR TIPO DE ATENCIÓN")
    total = cubo.agg(F.sum("atenciones")).first()[0]
    por_tipo = (
        cubo.groupBy("tipo_atencion").agg(F.sum("atenciones").alias("atenciones"))
        .withColumn("participacion_pct", F.round(F.col("atenciones") / total * 100, 2))
        .orderBy(F.desc("atenciones"))
    )
    por_tipo.show(truncate=False)
    pdf = por_tipo.toPandas()
    guardar_tabla(pdf, "p2_tipo_atencion.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(pdf["tipo_atencion"].str.title(), pdf["atenciones"], color=COLORES[:len(pdf)])
    for barra, pct in zip(barras, pdf["participacion_pct"]):
        ax.annotate(f"{pct:.1f}%", (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                    ha="center", va="bottom")
    ax.yaxis.set_major_formatter(millones)
    ax.set_title("Atenciones por tipo de atención (2009-2021)")
    ax.set_ylabel("Atenciones (millones)")
    guardar_grafica("p2_tipo_atencion.png")

    # Evolución anual por tipo (tabla dinámica con pivot)
    evolucion = (
        cubo.groupBy("anio").pivot("tipo_atencion").agg(F.sum("atenciones"))
        .orderBy("anio").toPandas().set_index("anio")
    )
    ax = evolucion.plot(figsize=(10, 5), marker="o", color=COLORES)
    ax.yaxis.set_major_formatter(millones)
    ax.set_title("Evolución anual por tipo de atención")
    ax.set_xlabel("Año")
    ax.set_ylabel("Atenciones (millones)")
    ax.grid(alpha=0.3)
    guardar_grafica("p2_evolucion_tipo_atencion.png")


def p3_departamentos(cubo):
    titulo("P3. ATENCIONES POR DEPARTAMENTO (énfasis en el Cesar)")
    ventana = Window.orderBy(F.desc("atenciones"))
    por_depto = (
        cubo.filter(F.col("cod_departamento") != "-1")  # se excluye "NO DEFINIDO"
        .groupBy("cod_departamento", "departamento_nombre", "region")
        .agg(F.sum("atenciones").alias("atenciones"))
        .withColumn("puesto", F.row_number().over(ventana))
        .orderBy("puesto")
    )
    por_depto.show(30, truncate=False)
    pdf = por_depto.toPandas()
    guardar_tabla(pdf, "p3_departamentos.csv")

    cesar = pdf[pdf["cod_departamento"] == DEPARTAMENTO_ENFASIS]
    if not cesar.empty:
        print(f"  El Cesar ocupa el puesto {int(cesar['puesto'].iloc[0])} de {len(pdf)} "
              f"con {int(cesar['atenciones'].iloc[0]):,} atenciones.")

    pdf = pdf.sort_values("atenciones")
    colores = [COLOR_ENFASIS if c == DEPARTAMENTO_ENFASIS else COLOR_PRINCIPAL
               for c in pdf["cod_departamento"]]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(pdf["departamento_nombre"], pdf["atenciones"], color=colores)
    ax.xaxis.set_major_formatter(millones)
    ax.set_title("Atenciones por departamento 2009-2021 (Cesar resaltado)")
    ax.set_xlabel("Atenciones (millones)")
    guardar_grafica("p3_departamentos.png")

    # Atenciones por región natural
    por_region = (
        cubo.groupBy("region").agg(F.sum("atenciones").alias("atenciones"))
        .orderBy(F.desc("atenciones")).toPandas()
    )
    guardar_tabla(por_region, "p3_regiones.csv")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(por_region["atenciones"], labels=por_region["region"], autopct="%1.1f%%",
           colors=COLORES, startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Distribución de atenciones por región natural")
    guardar_grafica("p3_regiones.png")


def p4_diagnosticos(spark, df):
    """Se usa Spark SQL para mostrar la misma consulta en lenguaje SQL."""
    titulo("P4. TOP 10 DIAGNÓSTICOS (Spark SQL)")
    df.createOrReplaceTempView("rips")

    # Los diagnósticos "NO DEFINIDO" se excluyen del ranking, pero se reporta
    # su peso porque es un hallazgo de calidad de datos importante.
    no_definido = spark.sql("""
        SELECT ROUND(100 * SUM(CASE WHEN diagnostico_nombre = 'NO DEFINIDO'
                                    THEN numero_atenciones ELSE 0 END)
                     / SUM(numero_atenciones), 2) AS pct_no_definido
        FROM rips
    """).first()[0]
    print(f"  Atenciones con diagnóstico NO DEFINIDO: {no_definido}% del total (se excluyen del top)\n")

    consulta = """
        SELECT cod_diagnostico,
               diagnostico_nombre,
               SUM(numero_atenciones) AS atenciones
        FROM rips
        WHERE diagnostico_nombre <> 'NO DEFINIDO' {filtro}
        GROUP BY cod_diagnostico, diagnostico_nombre
        ORDER BY atenciones DESC
        LIMIT 10
    """
    top_total = spark.sql(consulta.format(filtro=""))
    top_urgencias = spark.sql(consulta.format(filtro="AND tipo_atencion = 'URGENCIAS'"))

    print("Top 10 diagnósticos - todas las atenciones:")
    top_total.show(truncate=60)
    print("Top 10 diagnósticos - urgencias:")
    top_urgencias.show(truncate=60)

    fig, ejes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, datos, nombre, color in [(ejes[0], top_total, "Todas las atenciones", COLOR_PRINCIPAL),
                                     (ejes[1], top_urgencias, "Urgencias", COLOR_ENFASIS)]:
        pdf = datos.toPandas().iloc[::-1]
        guardar_tabla(pdf, f"p4_top_diagnosticos_{nombre.split()[0].lower()}.csv")
        etiquetas = (pdf["cod_diagnostico"] + " " + pdf["diagnostico_nombre"].str.slice(0, 38)).str.title()
        ax.barh(etiquetas, pdf["atenciones"], color=color)
        ax.xaxis.set_major_formatter(millones)
        ax.set_title(f"Top 10 diagnósticos - {nombre}")
        ax.set_xlabel("Atenciones (millones)")
    guardar_grafica("p4_top_diagnosticos.png")


def p5_impacto_pandemia(cubo):
    titulo("P5. IMPACTO DE LA PANDEMIA: 2020 FRENTE A 2019")
    comparacion = (
        cubo.filter(F.col("anio").isin(2019, 2020))
        .groupBy("tipo_atencion").pivot("anio", [2019, 2020]).agg(F.sum("atenciones"))
        .withColumn("variacion_pct", F.round((F.col("2020") - F.col("2019")) / F.col("2019") * 100, 1))
        .orderBy("variacion_pct")
    )
    comparacion.show(truncate=False)
    pdf = comparacion.toPandas()
    guardar_tabla(pdf, "p5_2019_vs_2020.csv")

    t2019, t2020 = pdf["2019"].sum(), pdf["2020"].sum()
    print(f"  Total 2019: {t2019:,}  |  Total 2020: {t2020:,}  |  "
          f"Variación: {(t2020 - t2019) / t2019 * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(pdf))
    ax.bar([i - 0.2 for i in x], pdf["2019"], width=0.4, label="2019", color=COLOR_PRINCIPAL)
    ax.bar([i + 0.2 for i in x], pdf["2020"], width=0.4, label="2020", color=COLOR_ENFASIS)
    for i, pct in zip(x, pdf["variacion_pct"]):
        ax.annotate(f"{pct:+.1f}%", (i, max(pdf["2019"][i], pdf["2020"][i])), ha="center", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pdf["tipo_atencion"].str.title())
    ax.yaxis.set_major_formatter(millones)
    ax.set_title("Atenciones 2019 vs 2020 por tipo de atención")
    ax.set_ylabel("Atenciones (millones)")
    ax.legend()
    guardar_grafica("p5_2019_vs_2020.png")


def capitulos_cie10(cubo):
    titulo("COMPLEMENTO. ATENCIONES POR CAPÍTULO CIE-10")
    por_capitulo = (
        cubo.groupBy("capitulo_cie10").agg(F.sum("atenciones").alias("atenciones"))
        .orderBy(F.desc("atenciones"))
    )
    por_capitulo.show(25, truncate=False)
    pdf = por_capitulo.toPandas()
    guardar_tabla(pdf, "capitulos_cie10.csv")
    pdf = pdf.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(pdf["capitulo_cie10"], pdf["atenciones"], color=COLOR_PRINCIPAL)
    ax.xaxis.set_major_formatter(millones)
    ax.set_title("Atenciones por capítulo de la CIE-10 (2009-2021)")
    ax.set_xlabel("Atenciones (millones)")
    guardar_grafica("capitulos_cie10.png")


def municipios_cesar_rdd(df):
    """
    Análisis con la API de RDD (transformaciones y acciones de bajo nivel):
    map -> filter -> reduceByKey -> takeOrdered / reduce / count.

    El filtro por departamento se hace primero con el DataFrame para que
    Parquet solo entregue las filas del Cesar (predicate pushdown); así el
    RDD trabaja con unos cientos de miles de filas y no con 38 millones.
    """
    titulo("COMPLEMENTO (RDD). TOP 10 MUNICIPIOS DEL CESAR")
    rdd = (
        df.filter(F.col("cod_departamento") == DEPARTAMENTO_ENFASIS)
        .select("municipio_nombre", "tipo_atencion", "numero_atenciones").rdd
        .map(lambda fila: (fila.municipio_nombre, fila.numero_atenciones))    # transformación: map
        .filter(lambda par: par[1] > 0)                                      # transformación: filter
        .reduceByKey(lambda a, b: a + b)                                     # transformación con shuffle
        .cache()                                                             # se reutiliza en 3 acciones
    )
    top = rdd.takeOrdered(10, key=lambda par: -par[1])                      # acción
    total_cesar = rdd.map(lambda par: par[1]).reduce(lambda a, b: a + b)    # acción
    print(f"  Municipios del Cesar con atenciones: {rdd.count()}")          # acción
    print(f"  Total de atenciones en el Cesar: {total_cesar:,}\n")
    for municipio, atenciones in top:
        print(f"  {municipio:<30} {atenciones:>14,}  ({atenciones / total_cesar * 100:5.1f}%)")

    if top:
        nombres = [m for m, _ in top][::-1]
        valores = [a for _, a in top][::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(nombres, valores, color=COLOR_ENFASIS)
        ax.xaxis.set_major_formatter(millones)
        ax.set_title("Top 10 municipios del Cesar por atenciones (2009-2021)")
        ax.set_xlabel("Atenciones (millones)")
        guardar_grafica("cesar_municipios.png")


def main():
    inicio = time.time()
    os.makedirs(DIR_GRAFICAS, exist_ok=True)
    os.makedirs(DIR_RESULTADOS, exist_ok=True)

    spark = crear_sesion()
    df, cubo = cargar(spark)

    estadisticas_generales(df)
    p1_atenciones_por_anio(cubo)
    p2_tipo_atencion(cubo)
    p3_departamentos(cubo)
    p4_diagnosticos(spark, df)
    p5_impacto_pandemia(cubo)
    capitulos_cie10(cubo)
    municipios_cesar_rdd(df)

    cubo.unpersist()
    print(f"\nGráficas en ./{DIR_GRAFICAS}  |  Tablas en ./{DIR_RESULTADOS}")
    print(f"Tiempo total: {(time.time() - inicio) / 60:.1f} minutos")
    spark.stop()


if __name__ == "__main__":
    main()
