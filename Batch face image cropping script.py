import os
import shutil
import pathlib
import collections
import cv2
import numpy as np

# Generar un nombre de directorio único para evitar sobreescribir
def f_directorio_salida():
    contador_val = 1
    directorio_salida = ""

    while True:
        directorio_salida = f"Faces Output ({contador_val})"
        if not pathlib.Path(directorio_salida).exists():
            break
        contador_val += 1

    return directorio_salida

# Agrupar archivos por carpeta
def f_agrupar_por_carpeta(directorio_capt, extensiones):
    archivos_por_carpeta = collections.defaultdict(list)
    
    for ext in extensiones:
        for entrada in pathlib.Path(directorio_capt).rglob(f'*.{ext}'):
            if entrada.is_file():
                carpeta_padre = entrada.parent
                archivos_por_carpeta[carpeta_padre].append(entrada)
    
    return archivos_por_carpeta

def cargar_modelo_dnn():
    """Carga el modelo DNN desde el directorio actual"""
    config_file = "deploy.prototxt"
    model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    
    if not os.path.exists(config_file) or not os.path.exists(model_file):
        raise FileNotFoundError("Archivos del modelo DNN no encontrados en el directorio actual")
    
    return cv2.dnn.readNetFromCaffe(config_file, model_file)

def detectar_rostros_dnn(img, net):
    """Detecta rostros usando el modelo DNN"""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    rostros = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Umbral de confianza
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            rostros.append((x1, y1, x2-x1, y2-y1))
    
    return rostros

def procesar_imagen(imagen_path, modelo_dnn):
    """Procesa la imagen usando el modelo DNN"""
    img = cv2.imread(str(imagen_path))
    if img is None:
        return []
    
    try:
        rostros = detectar_rostros_dnn(img, modelo_dnn)
    except Exception as e:
        print(f"Error al procesar {imagen_path}: {str(e)}")
        return []
    
    rostros_recortados = []
    for (x, y, w, h) in rostros:
        margen_w = int(w * 0.2)
        margen_h = int(h * 0.2)
        x1 = max(0, x - margen_w)
        y1 = max(0, y - margen_h)
        x2 = min(img.shape[1], x + w + margen_w)
        y2 = min(img.shape[0], y + h + margen_h)
        
        rostro = img[y1:y2, x1:x2]
        rostros_recortados.append(rostro)
    
    return rostros_recortados

def guardar_rostros(imagen_path, rostros, carpeta_destino):
    nombre_base = imagen_path.stem
    extension = imagen_path.suffix.lower()
    
    archivos_guardados = []
    
    if len(rostros) == 1:
        nombre_archivo = f"{nombre_base}{extension}"
        destino = carpeta_destino / nombre_archivo
        cv2.imwrite(str(destino), rostros[0])
        archivos_guardados.append(destino)
    elif len(rostros) > 1:
        for i, rostro in enumerate(rostros, 1):
            nombre_archivo = f"{nombre_base}({i}){extension}"
            destino = carpeta_destino / nombre_archivo
            cv2.imwrite(str(destino), rostro)
            archivos_guardados.append(destino)
    
    return archivos_guardados

def f_procesar_imagenes(directorio_capt, archivos_por_carpeta, directorio_salida):
    total_rostros = 0
    total_imagenes_procesadas = 0
    
    try:
        modelo_dnn = cargar_modelo_dnn()
        print("Modelo DNN cargado correctamente - usando detección de alta precisión")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 0, 0

    for carpeta_origen, archivos in archivos_por_carpeta.items():
        ruta_relativa = os.path.relpath(carpeta_origen, directorio_capt)
        carpeta_destino = pathlib.Path(directorio_salida) / ruta_relativa
        carpeta_destino.mkdir(parents=True, exist_ok=True)
        
        for imagen_path in archivos:
            rostros = procesar_imagen(imagen_path, modelo_dnn)
            if rostros:
                guardados = guardar_rostros(imagen_path, rostros, carpeta_destino)
                total_rostros += len(guardados)
                total_imagenes_procesadas += 1
    
    return total_rostros, total_imagenes_procesadas

def main():
    extensiones_soportadas = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
    
    print("=== Detector de Rostros con Modelo DNN ===")
    print("Asegúrate de tener estos archivos en el mismo directorio:")
    print("- deploy.prototxt")
    print("- res10_300x300_ssd_iter_140000_fp16.caffemodel\n")
    
    while True:
        # Directorio de entrada
        while True:
            directorio_capt = input("Ingrese el directorio con imágenes (o 'q' para salir): ").strip()
            if directorio_capt.lower() == 'q':
                return
            if pathlib.Path(directorio_capt).exists():
                break
            print("¡Directorio no encontrado!")
        
        # Obtener directorio de salida
        directorio_salida = f_directorio_salida()
        pathlib.Path(directorio_salida).mkdir()
        
        # Agrupar archivos por carpeta
        archivos_por_carpeta = f_agrupar_por_carpeta(directorio_capt, extensiones_soportadas)
        
        # Procesar imágenes
        total_rostros, total_imagenes = f_procesar_imagenes(
            directorio_capt, archivos_por_carpeta, directorio_salida)
        
        print("\n" + "="*50)
        print(f"Imágenes procesadas: {total_imagenes}")
        print(f"Rostros detectados: {total_rostros}")
        
        if total_rostros > 0:
            print(f"Resultados guardados en: {os.path.abspath(directorio_salida)}")
        else:
            shutil.rmtree(directorio_salida)
            print("No se detectaron rostros en las imágenes")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()