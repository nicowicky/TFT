prompt
Sanity check del script (end to end)
Uso Colab A100 GPU optimizar
escribir los function calss todos en una lista al final para ejecutarse manuealmente uno a uno.
todo debe quedar grabado para rehusarse.
formato sugerido
# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  tft_run_pipeline.py  –  orquestrador de llamadas paso a paso         ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# 0) IMPORTA EL MÓDULO CON LOS DEF
import tft_gpu_pipeline as tft

# ---------------------------- PASO 1 ------------------------------------
print("①  Cargar dataset …")
df = tft.load_dataset(tft.DATASET_PATH)
print("✔  filas:", len(df))

# ---------------------------- PASO 2 ------------------------------------
print("\n②  Crear (o cargar) series_raw y covars_raw …")
series_raw, covars_raw = tft.create_series(df)
print("✔  series_raw:", len(series_raw))

# ---------------------------- PASO 3 ------------------------------------
print("\n③  Filtrar series problemáticas …")
series, covars = tft.filter_invalid_splits(series_raw, covars_raw)
print("✔  series filtradas:", len(series))

# ---------------------------- PASO 4 ------------------------------------
print("\n④  Generar splits train/valid …")
tr_series, val_series, tr_cov, val_cov = tft.split_series_and_save(series, covars)
print("✔  longitud listas:", len(tr_series))

# ---------------------------- PASO 5 ------------------------------------
print("\n⑤  Muestreo aleatorio de 2 000 series para Optuna …")
SAMPLE_N = 2_000
if SAMPLE_N > len(tr_series): SAMPLE_N = len(tr_series)
rng = tft.random.Random(42)
idx = rng.sample(range(len(tr_series)), SAMPLE_N)
tr_s  = [tr_series[i]  for i in idx]
val_s = [val_series[i] for i in idx]
tr_c  = [tr_cov[i]     for i in idx]
val_c = [val_cov[i]    for i in idx]
print("✔  muestra:", len(tr_s))

# ---------------------------- PASO 6 ------------------------------------
print("\n⑥  Búsqueda Optuna (saltada si best_tft_params.pkl existe) …")
best_params = tft.hyperparameter_search(tr_s, val_s, tr_c, val_c, n_trials=30)
print("✔  best_params:", best_params)

# ---------------------------- PASO 7 ------------------------------------
print("\n⑦  Entrenamiento final con TODAS las series …")
model = tft.train_final_model(series_raw, covars_raw, best_params)
print("✔  modelo listo")

# ---------------------------- PASO 8 ------------------------------------
print("\n⑧  Extender covariables a meses 37-38 …")
covars_ext = tft.extend_covariates(covars_raw, n_months=2)
print("✔  covariables extendidas")

# ---------------------------- PASO 9 ------------------------------------
print("\n⑨  Predicción tn_scaled del mes +2 …")
preds_scaled = tft.predict_next_two(model, series_raw, covars_ext)
print("✔  filas predichas:", len(preds_scaled))

# ---------------------------- PASO 10 -----------------------------------
print("\n⑩  Des-escalar y generar CSV Kaggle …")
tft.prepare_kaggle_submission(preds_scaled, df, tft.KAGGLE_CSV)
print("✔  pipeline COMPLETO")
