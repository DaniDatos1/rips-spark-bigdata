# -*- coding: utf-8 -*-
"""
Tarea 3 - Procesamiento de Datos con Apache Spark (Big Data - UNAD, 202016911)
Autor: Daniel Andrés Mercado Trillos

SCRIPT 1 DE 2: CARGA, LIMPIEZA Y TRANSFORMACIÓN (procesamiento en batch)

Problema:
    ¿Cómo ha evolucionado la demanda de servicios de salud en Colombia entre
    2009 y 2021 según el tipo de atención, el departamento y el diagnóstico,
    y qué impacto tuvo la pandemia de 2020 en esa demanda?

Conjunto de datos:
    Registros Individuales de Prestación de Servicios de Salud (RIPS),
    Portal de Datos Abiertos de Colombia (datos.gov.co).
    ~38 millones de registros, 4,2 GB en CSV, almacenado en HDFS.

Qué hace este script:
    1. Carga el CSV desde HDFS con un esquema explícito.
    2. Perfila la calidad de los datos (nulos, valores inválidos, no definidos).
    3. Limpia: corrige la codificación de caracteres, normaliza espacios,
       elimina registros inválidos y consolida registros duplicados.
    4. Transforma: separa código y nombre, agrega región natural,
       capítulo CIE-10 y periodo frente a la pandemia.
    5. Guarda el resultado limpio en formato Parquet (particionado por año)
       para que el script 2 (análisis) lo lea de forma rápida.

Ejecución:
    spark-submit --driver-memory 2g 01_limpieza_rips.py
    (opcional) spark-submit 01_limpieza_rips.py <ruta_entrada> <ruta_salida>
"""

import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RUTA_ENTRADA = sys.argv[1] if len(sys.argv) > 1 else "hdfs://localhost:9000/Tarea3/rips.csv"
RUTA_SALIDA = sys.argv[2] if len(sys.argv) > 2 else "hdfs://localhost:9000/Tarea3/rips_limpio"

ANIO_MIN, ANIO_MAX = 2009, 2021

# Esquema explícito: evita que Spark recorra los 4 GB solo para inferir tipos
# (inferSchema=True obliga a una lectura adicional completa del archivo).
ESQUEMA = StructType([
    StructField("Departamento", StringType(), True),
    StructField("Municipio", StringType(), True),
    StructField("Año", IntegerType(), True),
    StructField("TipoAtencion", StringType(), True),
    StructField("Diagnostico", StringType(), True),
    StructField("NumeroAtenciones", LongType(), True),
])

# Región natural de Colombia según el código DANE del departamento.
REGIONES = {
    "08": "Caribe", "13": "Caribe", "20": "Caribe", "23": "Caribe", "44": "Caribe",
    "47": "Caribe", "70": "Caribe", "88": "Caribe",
    "05": "Andina", "11": "Andina", "15": "Andina", "17": "Andina", "25": "Andina",
    "41": "Andina", "54": "Andina", "63": "Andina", "66": "Andina", "68": "Andina",
    "73": "Andina",
    "19": "Pacífica", "27": "Pacífica", "52": "Pacífica", "76": "Pacífica",
    "50": "Orinoquía", "81": "Orinoquía", "85": "Orinoquía", "99": "Orinoquía",
    "18": "Amazonía", "86": "Amazonía", "91": "Amazonía", "94": "Amazonía",
    "95": "Amazonía", "97": "Amazonía",
}


def titulo(texto):
    """Imprime un encabezado visible en la consola para separar cada etapa."""
    print("\n" + "=" * 78)
    print(f"  {texto}")
    print("=" * 78)


def corregir_texto(texto):
    """
    Revierte la doble codificación del texto ("BogotÃ¡" -> "Bogotá").

    El archivo original se guardó en UTF-8, luego se leyó como Windows-1252
    y se volvió a guardar en UTF-8. Para revertirlo se recuperan los bytes
    originales (cada carácter vuelve a su byte en Windows-1252/Latin-1) y se
    decodifican nuevamente como UTF-8. Si el texto ya estaba bien, la
    decodificación falla y se devuelve sin cambios.
    """
    if texto is None:
        return None
    try:
        datos = bytearray()
        for caracter in texto:
            codigo = ord(caracter)
            if codigo < 256:
                datos.append(codigo)
            else:
                datos.extend(caracter.encode("cp1252"))
        return datos.decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def crear_sesion():
    """Crea la sesión de Spark con una configuración adecuada para una sola máquina."""
    spark = (
        SparkSession.builder
        .appName("RIPS - Limpieza y transformacion")
        # 200 particiones de shuffle (valor por defecto) es excesivo en una VM local.
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def cargar(spark):
    """Etapa 1: lectura del CSV desde HDFS."""
    titulo("1. CARGA DE DATOS DESDE HDFS")
    df = (
        spark.read
        .option("header", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .schema(ESQUEMA)
        .csv(RUTA_ENTRADA)
        # Se renombra "Año" para evitar la "ñ" en el nombre de la columna.
        .withColumnRenamed("Año", "Anio")
    )
    print(f"Fuente: {RUTA_ENTRADA}")
    print(f"Particiones de lectura: {df.rdd.getNumPartitions()}")
    df.printSchema()
    df.show(5, truncate=60)
    return df


def perfilar(df):
    """
    Etapa 2: perfil de calidad.
    Todas las métricas se calculan en UNA sola pasada sobre los datos
    (una única agregación), en lugar de lanzar un count() por cada métrica.
    """
    titulo("2. PERFIL DE CALIDAD DE LOS DATOS")
    metricas = df.agg(
        F.count(F.lit(1)).alias("total_registros"),
        *[F.sum(F.col(c).isNull().cast("int")).alias(f"nulos_{c}") for c in df.columns],
        F.sum((F.col("NumeroAtenciones") <= 0).cast("int")).alias("atenciones_no_positivas"),
        F.sum((~F.col("Anio").between(ANIO_MIN, ANIO_MAX)).cast("int")).alias("anio_fuera_de_rango"),
        F.sum(F.col("Departamento").startswith("-1").cast("int")).alias("departamento_no_definido"),
        F.sum(F.col("Diagnostico").contains("Ã").cast("int")).alias("diagnostico_mal_codificado"),
        F.sum("NumeroAtenciones").alias("total_atenciones"),
    ).collect()[0].asDict()

    for nombre, valor in metricas.items():
        print(f"  {nombre:<32} {valor:>16,}")
    return metricas


def construir_diccionario_codificacion(spark, df):
    """
    Construye una tabla pequeña "texto dañado -> texto corregido".

    En lugar de aplicar una función Python (UDF) a 38 millones de filas, se
    extraen los valores DISTINTOS de las columnas de texto (unos miles), se
    corrigen en el driver y luego se cruzan con un broadcast join, que Spark
    ejecuta en cada partición sin mover los datos grandes (sin shuffle).
    """
    distintos = (
        df.select(F.explode(F.array("Departamento", "Municipio", "Diagnostico")).alias("original"))
        .distinct()
        .rdd.map(lambda fila: fila.original)          # operación RDD: map
        .filter(lambda texto: texto is not None)      # operación RDD: filter
        .collect()
    )
    pares = [(t, corregir_texto(t)) for t in distintos]
    cambiados = sum(1 for original, nuevo in pares if original != nuevo)
    print(f"  Valores de texto distintos: {len(pares):,}  |  corregidos: {cambiados:,}")
    for original, nuevo in [p for p in pares if p[0] != p[1]][:5]:
        print(f"    '{original[:45]}'  ->  '{nuevo[:45]}'")
    return spark.createDataFrame(pares, ["original", "corregido"])


def limpiar(spark, df):
    """Etapa 3: limpieza de los datos."""
    titulo("3. LIMPIEZA")

    # 3.1 Corrección de codificación mediante broadcast join con el diccionario
    print("3.1 Corrección de la codificación de caracteres")
    diccionario = F.broadcast(construir_diccionario_codificacion(spark, df))
    for columna in ["Departamento", "Municipio", "Diagnostico"]:
        d = diccionario.withColumnRenamed("original", columna).withColumnRenamed("corregido", f"{columna}_ok")
        df = (
            df.join(d, on=columna, how="left")
            .withColumn(columna, F.coalesce(F.col(f"{columna}_ok"), F.col(columna)))
            .drop(f"{columna}_ok")
        )

    # 3.2 Normalización de espacios (incluye espacios no separables  )
    print("3.2 Normalización de espacios y mayúsculas")
    for columna in ["Departamento", "Municipio", "Diagnostico", "TipoAtencion"]:
        df = df.withColumn(columna, F.trim(F.regexp_replace(F.col(columna), r"[\s ]+", " ")))
    df = df.withColumn("TipoAtencion", F.upper("TipoAtencion"))

    # 3.3 Eliminación de registros inválidos
    print("3.3 Eliminación de nulos, años fuera de rango y atenciones <= 0")
    df = df.dropna(subset=["Departamento", "Municipio", "Anio", "TipoAtencion",
                           "Diagnostico", "NumeroAtenciones"])
    df = df.filter(F.col("Anio").between(ANIO_MIN, ANIO_MAX) & (F.col("NumeroAtenciones") > 0))

    # 3.4 Consolidación de duplicados.
    # No se usa dropDuplicates(): dos filas idénticas pueden corresponder a
    # subgrupos distintos de la fuente original y borrarlas perdería atenciones.
    # Se agrupan por su llave y se SUMAN las atenciones (no se pierde información).
    print("3.4 Consolidación de registros duplicados (suma de atenciones por llave)")
    df = (
        df.groupBy("Departamento", "Municipio", "Anio", "TipoAtencion", "Diagnostico")
        .agg(F.sum("NumeroAtenciones").alias("NumeroAtenciones"))
    )
    return df


def transformar(df):
    """Etapa 4: nuevas columnas para el análisis."""
    titulo("4. TRANSFORMACIÓN")
    patron = r"^\s*(-?[0-9A-Za-z]+)\s*-\s*(.*)$"

    # 4.1 Separar "código - nombre" en dos columnas
    for columna, prefijo in [("Departamento", "departamento"),
                             ("Municipio", "municipio"),
                             ("Diagnostico", "diagnostico")]:
        df = (
            df.withColumn(f"cod_{prefijo}", F.regexp_extract(columna, patron, 1))
            .withColumn(f"{prefijo}_nombre", F.regexp_extract(columna, patron, 2))
        )
    df = df.withColumn("diagnostico_nombre", F.upper("diagnostico_nombre"))
    print("4.1 Código y nombre separados para departamento, municipio y diagnóstico")

    # 4.2 Región natural a partir del código DANE (mapa literal, sin join)
    mapa_regiones = F.create_map(*[F.lit(x) for par in REGIONES.items() for x in par])
    df = df.withColumn("region", F.coalesce(mapa_regiones[F.col("cod_departamento")], F.lit("No definida")))
    print("4.2 Región natural asignada")

    # 4.3 Capítulo de la CIE-10 según la letra y el número del código
    letra = F.substring("cod_diagnostico", 1, 1)
    numero = F.substring("cod_diagnostico", 2, 2).cast("int")
    capitulo = (
        F.when(letra.isin("A", "B"), "Infecciosas y parasitarias")
        .when((letra == "C") | ((letra == "D") & (numero <= 48)), "Tumores")
        .when(letra == "D", "Sangre e inmunidad")
        .when(letra == "E", "Endocrinas y metabólicas")
        .when(letra == "F", "Trastornos mentales")
        .when(letra == "G", "Sistema nervioso")
        .when((letra == "H") & (numero <= 59), "Ojo")
        .when(letra == "H", "Oído")
        .when(letra == "I", "Sistema circulatorio")
        .when(letra == "J", "Sistema respiratorio")
        .when(letra == "K", "Sistema digestivo")
        .when(letra == "L", "Piel")
        .when(letra == "M", "Osteomuscular")
        .when(letra == "N", "Genitourinario")
        .when(letra == "O", "Embarazo, parto y puerperio")
        .when(letra == "P", "Perinatales")
        .when(letra == "Q", "Malformaciones congénitas")
        .when(letra == "R", "Síntomas y signos anormales")
        .when(letra.isin("S", "T"), "Traumatismos y envenenamientos")
        .when(letra.isin("V", "W", "X", "Y"), "Causas externas")
        .when(letra == "Z", "Factores de salud y contacto con servicios")
        .when(letra == "U", "Códigos especiales (COVID-19)")
        .otherwise("No definido")
    )
    df = df.withColumn("capitulo_cie10", capitulo)
    print("4.3 Capítulo CIE-10 asignado")

    # 4.4 Periodo frente a la pandemia
    df = df.withColumn(
        "periodo",
        F.when(F.col("Anio") < 2020, "Prepandemia (2009-2019)").otherwise("Pandemia (2020-2021)"),
    )
    print("4.4 Periodo prepandemia / pandemia asignado")

    return df.select(
        F.col("Anio").alias("anio"), "periodo", "region",
        "cod_departamento", "departamento_nombre",
        "cod_municipio", "municipio_nombre",
        F.col("TipoAtencion").alias("tipo_atencion"),
        "cod_diagnostico", "diagnostico_nombre", "capitulo_cie10",
        F.col("NumeroAtenciones").alias("numero_atenciones"),
    )


def guardar(spark, df, metricas_iniciales):
    """Etapa 5: escritura en Parquet y resumen final."""
    titulo("5. ALMACENAMIENTO EN PARQUET (HDFS)")
    # Parquet es columnar y comprimido: el análisis posterior solo lee las
    # columnas que necesita. Se particiona por año para filtrar más rápido.
    df.write.mode("overwrite").partitionBy("anio").parquet(RUTA_SALIDA)
    print(f"Datos limpios guardados en: {RUTA_SALIDA}")

    limpio = spark.read.parquet(RUTA_SALIDA)
    resumen = limpio.agg(
        F.count(F.lit(1)).alias("registros"),
        F.sum("numero_atenciones").alias("atenciones"),
    ).collect()[0]

    titulo("RESUMEN DE LA LIMPIEZA")
    print(f"  Registros originales:            {metricas_iniciales['total_registros']:>16,}")
    print(f"  Registros después de limpiar:    {resumen['registros']:>16,}")
    print(f"  Atenciones originales:           {metricas_iniciales['total_atenciones']:>16,}")
    print(f"  Atenciones después de limpiar:   {resumen['atenciones']:>16,}")
    limpio.printSchema()
    limpio.show(10, truncate=40)


def main():
    inicio = time.time()
    spark = crear_sesion()
    df = cargar(spark)
    metricas = perfilar(df)
    df = limpiar(spark, df)
    df = transformar(df)
    guardar(spark, df, metricas)
    print(f"\nTiempo total: {(time.time() - inicio) / 60:.1f} minutos")
    spark.stop()


if __name__ == "__main__":
    main()
