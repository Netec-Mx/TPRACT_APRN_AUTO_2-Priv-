# Sistema de predicción y clasificación de oportunidades comerciales

## Metadata

| Campo | Detalle |
|---|---|
| Duración | 105 minutos de laboratorio |
| Complejidad | Media |
| Nivel de Bloom | Aplicar / Analizar / Evaluar |
| Tecnología principal | Python, Jupyter Notebook y scikit-learn en Visual Studio Code |
| Modalidad | Práctica guiada individual |

## Descripción general

Durante esta práctica, los participantes integrarán regresión y clasificación para analizar oportunidades comerciales. Sobre un mismo conjunto de datos construirán una regresión lineal para estimar el valor esperado de referencia y una regresión logística para estimar la probabilidad de conversión y clasificar cada oportunidad según un umbral.

El trabajo se desarrollará en un notebook preparado con contexto por reto, código comentado, visualizaciones y celdas de interpretación. Los participantes retomarán la regresión del Capítulo 1 y profundizarán en sigmoide, costo logístico, descenso del gradiente, fronteras de decisión, sobreajuste y regularización. Finalmente, generarán ambas predicciones para oportunidades nuevas y explicarán cómo utilizarlas en una decisión comercial.

## Objetivos de aprendizaje

Al completar la práctica, será capaz de:

- Construir una regresión e interpretar el aporte de características adicionales al valor esperado;
- Relacionar puntuación lineal, sigmoide, costo y descenso del gradiente;
- Entrenar un clasificador y distinguir probabilidad, umbral y clase;
- Interpretar matriz de confusión y frontera de decisión en un caso comercial;
- Identificar sobreajuste y evaluar regularización sobre datos de validación;
- Integrar estimación monetaria y clasificación, justificando su uso y revisión humana.

## Prerrequisitos

### Conocimientos previos

Haber revisado el curso **Aprendizaje automático supervisado: Regresión y clasificación**, especialmente regresión lineal y logística, sigmoide, costo, gradiente, ingeniería de características, sobreajuste y regularización. La experiencia del Capítulo 1 se utiliza para retomar el flujo de modelado e interpretar errores.

### Acceso requerido

Equipo con Visual Studio Code, extensiones **Python** y **Jupyter** de Microsoft, y un intérprete Python de 64 bits preparado para formación. La familia de versiones prevista es **Python 3.11 a 3.13**. Use el entorno autorizado para el curso. La preparación detallada se encuentra en [requisitos del entorno]

## Entorno de laboratorio

### Materiales

Descargue y descomprima el proyecto antes de la sesión. Dentro de `Capitulo02/materiales/` encontrará:

| Archivo | Uso |
|---|---|
| `02_oportunidades_regresion_clasificacion.ipynb` | Código, explicaciones, experimentos y respuestas del participante. |
| `oportunidades_comerciales.csv` | 900 oportunidades para ambos modelos. |
| `requirements.txt` | Versiones de las dependencias directas. |
| `diccionario_datos.md` | Significado, unidades y momento de disponibilidad de cada columna. |

El notebook guarda gráficos y tablas en `materiales/resultados/` durante la ejecución. Complete sus interpretaciones en las celdas ya preparadas del notebook y guárdelo con **Ctrl+S**.

### Caso de trabajo

Un equipo comercial prepara el seguimiento de sus oportunidades. Cuenta con tamaño del cliente, interacciones, días transcurridos, servicios solicitados, monto nominal, reuniones con decisor, propuesta formal y antigüedad de la relación, registrados al corte de evaluación.

**Dos resultados:** `valor_esperado_millones_cop` representa una expectativa económica de referencia y `convertida` representa el desenlace posterior, 0 o 1. El caso distingue el monto nominal de la oportunidad, el valor esperado y la venta individual concretada. 

## Procedimiento paso a paso

### Preparación — 5 minutos

1. Abra **Visual Studio Code > File > Open Folder** y seleccione `Capitulo02/materiales`.
2. Abra `02_oportunidades_regresion_clasificacion.ipynb` desde **Explorer**. Confirme que el CSV y `requirements.txt` aparecen en la misma carpeta.
3. Seleccione **Select Kernel > Python Environments** y elija el Python preparado para el curso. Si aparece la lista de recientes, seleccione primero **Select Another Kernel**.
4. Si VS Code solicita **ipykernel**, pulse **Install** para habilitar la ejecución de celdas en ese intérprete.
5. En **0A. Instalar dependencias**, ejecute la celda con **Run Cell** o **Shift+Enter**:

   ```python
   %pip install -r requirements.txt --progress-bar on
   ```

6. Observe la salida debajo de la celda y espere la finalización. Reinicie el kernel cuando la herramienta lo solicite. Ejecute **0B. Importar herramientas y cargar el caso**.
7. Confirme **900 registros, 11 columnas**, el nombre del CSV y el mensaje de preparación verificada.

> **Preparación previa:** la instalación inicial de Python, extensiones y paquetes se completa una sola vez; su descarga depende de la red corporativa. Los cinco minutos de la sesión corresponden a la comprobación de un equipo preparado.

**Resultado esperado:** kernel activo, librerías importadas y datos disponibles.

**Verificación:** confirme la ruta del intérprete mostrada por la celda y la carga de las 900 filas.

---

### Reto 1 — Explorar el caso y separar los datos — 8 minutos

**Objetivo:** Distinguir los dos objetivos y preparar una evaluación separada del entrenamiento.

1. Ejecute las dos celdas de **1. Explorar el caso y separar los datos**.

2. Compruebe la partición: 540 filas de entrenamiento, 180 de validación y 180 de prueba.

3. Revise columnas y proporción de clases. Distinga características al corte de evaluación de objetivos.

4. Lea las indicaciones del gráfico de dispersión y localice oportunidades de monto similar con expectativas diferentes.

5. Complete la celda **Interpretación del problema integrado** incluida al final de este reto en el notebook.

**Resultado esperado:** Las dos salidas y la función de cada partición quedan identificadas.

**Verificación:** La suma de las particiones es 900 y el gráfico corresponde a entrenamiento.

---

### Reto 2 — Estimar el valor esperado mediante regresión — 12 minutos

**Objetivo:** Reutilizar el flujo de regresión para estimar el valor económico y comparar representaciones.

1. Ejecute las celdas de **2. Estimar el valor esperado mediante regresión**.

2. Observe cómo `crear_caracteristicas` genera intensidad comercial, valor por servicio e interacción con propuesta.

3. Compare MAE, RMSE y R² en **validación**. Identifique la alternativa seleccionada por menor MAE.

4. Lea el signo y la unidad de los coeficientes estandarizados de la regresión base.

5. Complete la celda **Interpretación de la regresión comercial** incluida al final de este reto en el notebook.

**Resultado esperado:** Dos regresiones comparadas y una selección basada en MAE de validación.

**Verificación:** La tabla contiene base y ampliada; las unidades de MAE/RMSE son millones de COP.

---

### Reto 3 — Experimentar con sigmoide, costo y gradiente — 17 minutos

**Objetivo:** Relacionar la probabilidad con la puntuación, la pérdida y las actualizaciones del gradiente.

1. En **3. Experimentar con sigmoide, costo y gradiente**, ejecute **3A** y lea las dos gráficas.

2. Compare z = -4, 0 y 4. Observe la pérdida de asignar p = 0.1 a una venta concretada frente a p = 0.9.

3. Ejecute la celda de funciones de **3B**. Relacione los comentarios de `X @ w`, `X.T @ error` y `error.mean()` con la explicación previa.

4. Ejecute **3C** con `ALPHAS = [0.01, 0.10, 0.50]` y observe costo inicial, costo final y estabilidad.

5. Cambie la lista a `ALPHAS = [0.01, 0.10, 0.30, 0.50]`. Ejecute nuevamente esa celda y compare. Conserve la observación de su experimento.

6. Complete la celda **Interpretación de sigmoide, costo y gradiente** incluida al final de este reto en el notebook.

**Resultado esperado:** Relación observable entre sigmoide, pérdida y optimización.

**Verificación:** z = 0 produce p = 0.5 y las ejecuciones predeterminadas reducen el costo.

---

### Reto 4 — Entrenar y evaluar la regresión logística — 17 minutos

**Objetivo:** Evaluar probabilidades, clases y consecuencias de cambiar el umbral.

1. Ejecute **4A** en **4. Entrenar y evaluar la regresión logística**.

2. Lea la tabla de métricas y compárela con la referencia de clase mayoritaria.

3. Ejecute **4B** con `UMBRAL_EXPERIMENTO = 0.50`. Lea la matriz por filas reales y columnas predichas.

4. Cambie a `0.35` y luego a `0.65`; vuelva a ejecutar **4B** cada vez. Compare VP, FP, FN, VN, precisión, recall y cantidad de probables.

5. Observe que cambió la regla de clase, mientras las probabilidades se conservaron.

6. Complete la celda **Interpretación de probabilidades y errores** incluida al final de este reto en el notebook.

**Resultado esperado:** Clasificador evaluado y efectos del umbral explicados con conteos reales.

**Verificación:** VP + FP + FN + VN suman 180 para cada umbral.

---

### Reto 5 — Interpretar la frontera de decisión — 10 minutos

**Objetivo:** Interpretar la frontera en dos dimensiones y distinguirla del clasificador principal.

1. Ejecute **5. Interpretar la frontera de decisión**.

2. Localice los ejes, la escala de probabilidad, la línea y los marcadores reales.

3. Identifique un punto en la región contraria a su desenlace y otro cercano a la línea.

4. Cambie `UMBRAL_FRONTERA` de `0.50` a `0.65` y vuelva a ejecutar la celda. Observe el desplazamiento.

5. Complete la celda **Interpretación de la frontera de decisión** incluida al final de este reto en el notebook.

**Resultado esperado:** Lectura de regiones de clasificación y de la sensibilidad al umbral.

**Verificación:** El gráfico identifica interacciones y días; la línea corresponde al umbral seleccionado.

---

### Reto 6 — Detectar sobreajuste y aplicar regularización — 18 minutos

**Objetivo:** Identificar sobreajuste y seleccionar regularización mediante validación.

1. Ejecute **6A** en **6. Detectar sobreajuste y aplicar regularización**. Lea la tabla de la muestra de 50 filas.

2. Compare el extremo `C = 1_000_000`, valores intermedios y `C = 0.0001`. Revise exactitud, log loss y norma de pesos.

3. Ejecute el gráfico de curvas. Relacione el eje logarítmico con la fuerza de regularización.

4. Ejecute **6B** para comparar los valores de C sobre las 540 filas y ocho entradas del modelo comercial.

5. Observe el C seleccionado por menor log loss de validación y la tabla de umbrales recalculada para ese modelo.

6. Fije `UMBRAL_FINAL` en `0.35`, `0.50` o `0.65` según su interpretación de los errores y ejecute nuevamente **6B** antes del Reto 7.

7. Complete la celda **Interpretación de sobreajuste y regularización** incluida al final de este reto en el notebook.

**Resultado esperado:** Evidencia de sobreajuste y configuración comercial fijada usando validación.

**Verificación:** El modelo comercial utiliza ocho entradas y 540 filas; el experimento polinómico usa dos entradas y 50 filas.

---

### Reto 7 — Evaluar e integrar predicciones de nuevas oportunidades — 13 minutos

**Objetivo:** Medir el resultado final y combinar las dos salidas para nuevas oportunidades.

1. Ejecute **7A** en **7. Evaluar e integrar predicciones de nuevas oportunidades**. Lea las métricas de prueba.

2. Ejecute **7B** y observe para A, B y C el valor esperado, la probabilidad, la clase y el campo de revisión humana.

3. Ejecute **7C**. Compare el escenario C con la copia que aumenta cuatro interacciones y mantiene el resto de los valores.

4. Para introducir otra oportunidad, edite una fila de `nuevas_oportunidades` en **7B**, conserve las columnas y ejecute **7B** y **7C**. Los campos expresan conteos, días, meses y millones de COP.

5. Abra `materiales/resultados` desde Explorer para comprobar las exportaciones. Mantenga sus explicaciones dentro del notebook.

6. Complete la celda **Interpretación de las predicciones integradas** incluida al final de este reto en el notebook.

**Resultado esperado:** Predicciones de ambos modelos, comparación de un cambio controlado y salidas exportadas.

**Verificación:** Existen tres resultados de oportunidades nuevas, dos filas en la comparación del escenario C y métricas de prueba.

---

### Cierre — 5 minutos

1. Ejecute la celda de **8. Cierre y verificación**.
2. Complete **Conclusiones de la práctica** en el notebook.
3. Guarde con **Ctrl+S** y explique su decisión de seguimiento en una frase sustentada en los resultados.

**Resultado esperado:** notebook ejecutado con interpretaciones y conclusiones propias.

## Validación y pruebas finales

- [ ] El notebook carga 900 oportunidades y conserva el archivo fuente.
- [ ] Entrenamiento, validación y prueba contienen 540, 180 y 180 filas distintas.
- [ ] La regresión compara las dos representaciones con métricas de validación.
- [ ] Se relacionaron sigmoide, costo y gradiente, y se probó una tasa adicional.
- [ ] Se interpretaron probabilidad, matriz de confusión y umbral.
- [ ] Se observó la frontera de dos entradas y el efecto de cambiar el umbral.
- [ ] Se identificó sobreajuste y se seleccionó C mediante validación.
- [ ] Se evaluó una configuración final sobre prueba y se generaron las dos salidas para nuevos casos.
- [ ] Se completaron las siete interpretaciones y las conclusiones en el notebook.

## Solución de problemas

| Situación | Acción |
|---|---|
| El notebook solicita kernel | Abra **Select Kernel**, elija **Python Environments** y el intérprete del curso. Acepte instalar `ipykernel` cuando se solicite. |
| `requirements.txt` no se encuentra | Abra `Capitulo02/materiales` como carpeta de trabajo. Reinicie el kernel y ejecute la instalación. |
| La instalación se detiene por red o certificados | Conserve el mensaje y solicite al equipo de soporte el acceso al repositorio autorizado. Utilice el entorno previamente preparado para continuar. |
| `ModuleNotFoundError` | Verifique el kernel seleccionado y ejecute `%pip install -r requirements.txt --progress-bar on` en ese notebook. |
| `NameError` tras reiniciar | Ejecute desde **0B** hasta la sección actual para reconstruir variables y modelos. |
| Aparece `ConvergenceWarning` | Compruebe el escalamiento y los parámetros indicados. Restaure la configuración base o aumente `max_iter`, ejecute y compare. |
| Se modificó C o una característica | Vuelva a ejecutar desde esa sección hasta las predicciones para mantener resultados consistentes. |
| Una predicción monetaria queda fuera de rango | Revise `revision_humana`, el monto nominal y las entradas. Registre la limitación del modelo antes de utilizar la estimación. |
| Las métricas difieren de la referencia | Compare semilla, versiones, datos y parámetros. Los experimentos del participante pueden cambiar las salidas. |

## Limpieza del entorno

1. Guarde el notebook con sus respuestas y compruebe los archivos de `materiales/resultados/`.
2. Conserve el CSV original y el notebook trabajado como referencia.
3. Cierre Visual Studio Code al finalizar y siga las indicaciones de la organización para conservar los materiales.

## Resumen

La práctica integró estimación monetaria y clasificación en un mismo caso comercial. La regresión permitió reutilizar el flujo conocido; la sigmoide, el costo logístico y el gradiente explicaron el aprendizaje del clasificador. La comparación de umbrales, fronteras y regularización permitió analizar decisiones y generalización. Las predicciones finales se interpretan junto con su evidencia y los controles de revisión humana.

## Recursos adicionales

- [Curso de Coursera](https://www.coursera.org/learn/machine-learning).
- [Regresión logística: documentación 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [Umbral de clasificación](https://scikit-learn.org/stable/modules/classification_threshold.html).
- [Separación de datos y pipelines](https://scikit-learn.org/stable/common_pitfalls.html).
- [Kernels en VS Code](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management).
