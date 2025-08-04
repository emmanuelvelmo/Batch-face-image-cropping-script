import collections # defaultdict para agrupar archivos por carpeta
import pathlib # Manejo moderno de rutas de archivos y directorios
import cv2 # OpenCV: lectura de imágenes, detección de rostros, guardado
import numpy # Operaciones con arrays numéricos (coordenadas de rostros)

# FUNCIONES
# Guarda las imágenes de rostros recortados con nomenclatura apropiada
def guardar_rostros(dir_imagen, rostros_imgs, carpeta_destino, directorio_salida):
    # Crear directorio de salida si no existe
    pathlib.Path(directorio_salida).mkdir(exist_ok = True)

    # Extraer nombre base y extensión del archivo original
    nombre_archivo = dir_imagen.stem
    extension_archivo = dir_imagen.suffix.lower()
    
    cont_rostros = 0 # Contador de archivos guardados
    
    # Lógica de nomenclatura según cantidad de rostros detectados
    if len(rostros_imgs) == 1:
        # Un solo rostro: mantener nombre original
        cv2.imwrite(str(carpeta_destino / f"{nombre_archivo}{extension_archivo}"), rostros_imgs[0]) # Guardar imagen de rostro
        
        cont_rostros += 1
    elif len(rostros_imgs) > 1:
        # Múltiples rostros: agregar numeración secuencial
        for indice_val, rostro_iter in enumerate(rostros_imgs, 1):
            cv2.imwrite(str(carpeta_destino / f"{nombre_archivo}({indice_val}){extension_archivo}"), rostro_iter) # Guardar imagen de rostro
            
            cont_rostros += 1
    
    return cont_rostros

# Detecta rostros en una imagen usando red neuronal DNN con umbral de confianza
def coordenadas_rostros(imagen_val, modelo_dnn):
    alto_val, ancho_val = imagen_val.shape[:2] # Obtener dimensiones de la imagen original
    
    # Crear blob: redimensionar imagen a 300x300 y normalizar valores de píxeles
    blob_val = cv2.dnn.blobFromImage(cv2.resize(imagen_val, (300, 300)), 1.0, (300, 300), [104, 117, 123])
    
    modelo_dnn.setInput(blob_val) # Establecer blob como entrada de la red neuronal
    
    detecciones_val = modelo_dnn.forward() # Ejecutar inferencia y obtener detecciones
    
    rostros_coords = [] # Lista para almacenar coordenadas de rostros válidos
    
    # Procesar cada detección encontrada por la red neuronal
    for iter_val in range(detecciones_val.shape[2]):
        # Extraer nivel de confianza de la detección actual
        umbral_confianza = detecciones_val[0, 0, iter_val, 2]
        
        # Filtrar detecciones con confianza mayor al umbral (70%)
        if umbral_confianza > 0.7: # Umbral de confianza para reducir falsos positivos
            # Extraer coordenadas del rostro y escalar a dimensiones originales
            caja = detecciones_val[0, 0, iter_val, 3:7] * numpy.array([ancho_val, alto_val, ancho_val, alto_val])
            
            # Convertir coordenadas flotantes a enteros
            (x1, y1, x2, y2) = caja.astype("int")
            
            # Asegurar que las coordenadas estén dentro de los límites de la imagen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ancho_val, x2), min(alto_val, y2)
            
            # Convertir formato de coordenadas (x1,y1,x2,y2) a (x, y, ancho, alto)
            rostros_coords.append((x1, y1, x2-x1, y2-y1))
    
    return rostros_coords

# Procesa una imagen individual: carga, detecta rostros y recorta con márgenes
def recortar_rostros(ruta_imagen, modelo_dnn):
    # Cargar imagen desde archivo usando OpenCV
    imagen_val = cv2.imread(str(ruta_imagen))
    
    try:
        # Ejecutar detección de rostros en la imagen
        rostros_coordenadas = coordenadas_rostros(imagen_val, modelo_dnn)
    except Exception as e:
        return []

    # Lista para almacenar imágenes recortadas de rostros
    rostros_imagenes = []
    
    # Recortar cada rostro detectado con márgenes adicionales
    for (x, y, w, h) in rostros_coordenadas:
        # Calcular márgenes (20% del tamaño del rostro para mejor encuadre)
        margen_horizontal = int(w * 0.2)
        margen_vertical = int(h * 0.2)
        
        # Calcular coordenadas expandidas con márgenes
        x1 = max(0, x - margen_horizontal)
        y1 = max(0, y - margen_vertical)
        x2 = min(imagen_val.shape[1], x + w + margen_horizontal)
        y2 = min(imagen_val.shape[0], y + h + margen_vertical)
        
        # Recortar región del rostro con márgenes de la imagen original
        rostro_recortado = imagen_val[y1:y2, x1:x2]
        
        # Agregar imagen al array de imágenes de rostros
        rostros_imagenes.append(rostro_recortado)
    
    return rostros_imagenes

# Procesamiento de imágenes y guardado de resultados
def procesar_directorio_imagenes(directorio_entrada, lista_carpetas_archivos, directorio_salida, modelo_dnn):
    # Contadores
    total_rostros_extraidos = 0
    total_imagenes_procesadas = 0
    
    # Procesar cada carpeta y sus archivos de imagen
    for iter_carpeta, lista_archivos in lista_carpetas_archivos.items():
        # Generar ruta relativa para mantener estructura de directorios
        carpeta_destino = pathlib.Path(directorio_salida) / pathlib.Path(iter_carpeta).relative_to(pathlib.Path(directorio_entrada))
        
        # Crear carpeta de destino si no existe
        carpeta_destino.mkdir(parents = True, exist_ok = True)
        
        # Procesar cada imagen individual de la carpeta actual
        for dir_imagen in lista_archivos:
            # Extraer rostros de la imagen actual
            rostros_encontrados = recortar_rostros(dir_imagen, modelo_dnn)
            
            # Guardar rostros si se encontró alguno
            if rostros_encontrados:
                total_rostros_extraidos += guardar_rostros(dir_imagen, rostros_encontrados, carpeta_destino, directorio_salida)
                
            total_imagenes_procesadas += 1
    
    return total_imagenes_procesadas, total_rostros_extraidos

# Organiza archivos de imagen agrupándolos por carpeta (búsqueda recursiva)
def agrupar_archivos_carpetas(directorio_origen, extensiones_lista):
    # Diccionario para carpetas y archivos de la misma
    dicc_carpetas_archivos = collections.defaultdict(list)
    
    # Buscar archivos para cada extensión de imagen soportada
    for extension_val in extensiones_lista:
        # Búsqueda recursiva en todas las subcarpetas usando patrón glob
        for archivo_iter in pathlib.Path(directorio_origen).rglob(f'*.{extension_val}'):
            # Verificar que sea un archivo válido y no un directorio
            if archivo_iter.is_file():
                carpeta_contenedora = archivo_iter.parent
                
                dicc_carpetas_archivos[carpeta_contenedora].append(archivo_iter)
    
    return dicc_carpetas_archivos

# Inicializa el modelo de red neuronal DNN para detección de rostros
def cargar_modelo_dnn():
    # Buscar el primer archivo prototxt en el directorio actual (no recursivo)
    archivo_configuracion = None
    
    for archivo_iter in pathlib.Path('.').glob('*.prototxt'):
        archivo_configuracion = str(archivo_iter)
        
        break
    
    # Buscar el primer archivo caffemodel solo en el directorio actual (no recursivo)
    archivo_pesos_modelo = None
    
    for archivo_iter in pathlib.Path('.').glob('*.caffemodel'):
        archivo_pesos_modelo = str(archivo_iter)
        
        break
    
    # Cargar y retornar modelo
    return cv2.dnn.readNetFromCaffe(archivo_configuracion, archivo_pesos_modelo)

# PUNTO DE PARTIDA
# Lista de formatos de imagen soportados por OpenCV
extensiones_lista = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'gif', 'heic']

try:
    # Cargar archivos para modelo
    modelo_dnn = cargar_modelo_dnn()
    
    # Bucle principal del programa
    while True:
        # Solicitar directorio de entrada
        while True:
            directorio_entrada = input("Enter directory: ").strip('"\'')
            
            # Verificar que el directorio exista
            if not pathlib.Path(directorio_entrada).exists():
                print("Wrong directory\n")
            else:
                break
        
        # Generar nombre para directorio de salida
        directorio_salida = f"{pathlib.Path(directorio_entrada).name} (output)"

        # Generar lista de directorios de imágenes en carpeta de entrada
        lista_carpetas_archivos = agrupar_archivos_carpetas(directorio_entrada, extensiones_lista)
        
        # Ejecutar detección, extracción y guarado de rostros
        total_imagenes, total_rostros = procesar_directorio_imagenes(directorio_entrada, lista_carpetas_archivos, directorio_salida, modelo_dnn)
        
        # Mostrar separador visual para resultados
        print("-" * 36)
        
        # Mostrar resultados de procesamiento
        if total_imagenes > 0:
            print(f"Processed images: {total_imagenes}")
            
            # Mostrar cantidad de rostros sólo si se encontró alguno
            if total_rostros > 0:
                print(f"Processed faces: {total_rostros}")
            else:
                print("No faces found")
        else:
            print("No images found")
        
        print("-" * 36 + "\n")
except Exception as e:
    print("No model found")
        
    # Detener el programa
    input()
