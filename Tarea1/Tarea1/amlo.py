import pandas as pd
import glob
import os
import re

# 1. Rutas
path_raiz = "./conferencias_matutinas_amlo" 
output_dir = "./corpus_final_txt"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Buscamos los archivos (excluyendo carpetas de participantes para no duplicar)
archivos_csv = sorted(glob.glob(os.path.join(path_raiz, "**/*.csv"), recursive=True))
archivos_finales = [f for f in archivos_csv if "csv_por_participante" not in f]

print(f"Transformando {len(archivos_finales)} archivos a texto plano...")

for archivo in archivos_finales:
    try:
        df = pd.read_csv(archivo)
        columna_texto = 'Texto' if 'Texto' in df.columns else 'Párrafo'
        
        if columna_texto in df.columns:
            # Unimos los párrafos
            texto_dia = "\n".join(df[columna_texto].astype(str))
            
            # --- Pseudo-limpieza ---
            # Removemos etiquetas de diálogo comunes o ruidos de transcripción
            texto_dia = re.sub(r'PRESIDENTE.*?:', '', texto_dia) # Quita el nombre del presidente al inicio
            
            # Generamos un nombre limpio basado en el archivo original
            # El original suele ser 'mananera_04_12_2018.csv'
            nombre_base = os.path.basename(archivo).replace('.csv', '.txt')
            
            with open(os.path.join(output_dir, nombre_base), 'w', encoding='utf-8') as f:
                f.write(texto_dia)
    except Exception as e:
        print(f"Error procesando {archivo}: {e}")

print(f"¡Hecho! Carpeta '{output_dir}' creada con los archivos .txt listos.")