# Tarea 3: Filtrado Frecuencial Notch y Deconvolución Regularizada
**Curso:** IEE2714 — Fundamentos de Procesamiento Digital de Imágenes  

---

## Descripción del Proyecto

Este repositorio contiene la implementación, experimentación y análisis cuantitativo de dos problemas fundamentales en el dominio frecuencial de Fourier:
1. **Eliminación de interferencias armónicas periódicas (ruido en franjas)** mediante filtros selectivos Notch (Ideales y Gaussianos).
2. **Restauración de imágenes degradadas por desenfoque (PSF) y ruido aditivo** mediante la solución cerrada de Wiener y el método variacional de Tikhonov Iterativo (Descenso de Gradiente Proyectado).

---

## Estructura del Repositorio

```text
.
├── p1_implementacion.py     # Funciones de FFT, construcción de filtros Notch y padding
├── p2_implementacion.py     # Métodos de deconvolución de Wiener, Tikhonov e indicadores RMSE
├── P1_notebook.ipynb        # Experimentos de la Pregunta 1 (Filtrado Notch y Padding)
├── P2_notebook.ipynb        # Experimentos de la Pregunta 2 (Wiener, Tikhonov y Sensibilidad PSF)
├── tarea3_files/            # Archivos de entrada (P1.CR2, cameraman.jpg, psf.png)
├── results/                 # Gráficos y reconstrucciones generadas
└── README.md                # Descripción del proyecto