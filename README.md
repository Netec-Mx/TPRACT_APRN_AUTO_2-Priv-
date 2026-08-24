# Taller Práctico Aprendizaje automático supervisado: Regresión y clasificación parte 2

Este taller práctico integrador permite aplicar los principales contenidos desarrollados a lo largo del curso mediante la construcción de una solución de aprendizaje automático supervisado orientada a un caso empresarial. Los participantes utilizarán un único conjunto de datos para desarrollar un modelo de regresión que estime el valor esperado de una oportunidad comercial y un modelo de clasificación que determine su probabilidad de conversión. La práctica integra regresión lineal, regresión logística, función sigmoide, función de costo, descenso del gradiente, fronteras de decisión, ingeniería de características, sobreajuste y regularización. Como resultado, los participantes crearán un notebook funcional capaz de recibir los datos de una nueva oportunidad y generar tanto una estimación numérica como una clasificación binaria.

## Estructura

- `CapituloXX/README.md`: guía de laboratorio por capítulo.

## Lista de laboratorios

### Capítulo 1

- [Sistema de predicción y clasificación de oportunidades comerciales](Capitulo01/README.md#sistema-de-predicción-y-clasificación-de-oportunidades-comerciales)
  - Descripción: Desarrollar una solución de aprendizaje supervisado que estime el valor esperado de una oportunidad y determine si tiene alta o baja probabilidad de convertirse en una venta. El ejercicio integra regresión lineal, regresión logística, función sigmoide, función de costo, descenso del gradiente, fronteras de decisión, ingeniería de características, sobreajuste y regularización. La actividad se desarrolla de manera guiada mediante un notebook base, de forma que el participante pueda concentrarse en ejecutar, ajustar e interpretar los modelos sin tener que construir desde cero los componentes de mayor complejidad. Se proporciona un único conjunto de datos con características como tamaño del cliente, interacciones comerciales, duración del proceso, número de servicios solicitados y valor de la oportunidad. En Visual Studio Code, los participantes trabajan sobre un notebook de Jupyter previamente estructurado y construyen primero un modelo de regresión para predecir el valor esperado y después un modelo de regresión logística para clasificar la oportunidad como probable o no probable. A partir de bloques de código preparados, comparan diferentes características, observan el comportamiento de la función sigmoide y las fronteras de decisión, identifican señales de sobreajuste y aplican regularización. Finalmente, utilizan el modelo para ingresar los datos de una nueva oportunidad y generar ambas predicciones. Se utilizan Pandas y Scikit-learn para el procesamiento y modelado de los datos, Matplotlib o Seaborn para las visualizaciones y Copilot Chat Standard como apoyo para comprender el código, interpretar resultados y resolver errores durante la práctica.
  - Duración estimada: 105 min

## Flujo de colaboración

- Trabajar en `changes_course`.
- Crear Pull Request hacia `main`.
- Merge por `Squash and merge`.
