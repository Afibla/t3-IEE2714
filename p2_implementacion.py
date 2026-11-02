import numpy as np

def preparar_psf(psf, shape_objetivo):
    # Normalizar la PSF para que su suma sea 1
    psf = psf.astype(np.float64)
    suma_psf = np.sum(psf)
    if suma_psf > 0:
        psf = psf / suma_psf

    # Extender la PSF con 0 padding al tamaño de la imagen
    M, N = shape_objetivo
    m, n = psf.shape
    
    psf_padded = np.zeros((M, N), dtype=np.float64)
    psf_padded[:m, :n] = psf
    
    # Desplazar la PSF para que su centro quede en la posición (0,0) de la FFT
    centro_m, centro_n = m // 2, n // 2
    psf_padded = np.roll(psf_padded, -centro_m, axis=0)
    psf_padded = np.roll(psf_padded, -centro_n, axis=1)

    # Calcular la FFT 2D de la PSF
    H = np.fft.fft2(psf_padded)
    return psf_padded, H

# Simula y = h*x + eta mediante convolución circular y ruido gaussiano
def simular_degradacion(x, H, sigma_ruido, seed=42):
    rng = np.random.default_rng(seed)

    X = np.fft.fft2(x)
    y_limpia = np.real(np.fft.ifft2(H * X))

    ruido = rng.normal(0, sigma_ruido, size=x.shape)
    y = y_limpia + ruido

    return y

# Restaura la imagen y usando la solución de Wiener X_hat
def deconvolucion_wiener(y, H, K):
    Y = np.fft.fft2(y)
    H_conj = np.conj(H)
    H_abs_sq = np.abs(H)**2
    
    W = H_conj / (H_abs_sq + K)
    X_hat = W * Y
    
    x_hat = np.real(np.fft.ifft2(X_hat))
    return np.clip(x_hat, 0.0, 1.0)

def deconvolucion_tikhonov_iterativo(
    y, H, lambda_reg, tau=None,
    max_iter=200, tol=1e-6, x_true=None
    ):

    if lambda_reg < 0:
        raise ValueError("lambda_reg debe ser no negativo.")

    Y = np.fft.fft2(y)

    H_conj = np.conj(H)
    H_abs_sq = np.abs(H)**2

    # Calcular paso de aprendizaje seguro
    L = np.max(H_abs_sq) + lambda_reg

    if L <= 0:
        raise ValueError("El problema no tiene un gradiente válido.")

    if tau is None:
        tau = 0.9 / L
    elif tau <= 0 or tau > 1.0 / L:
        raise ValueError(f"tau debe estar en (0, {1.0/L:.6f}].")

    # Inicialización
    x_k = np.clip(y.copy(), 0.0, 1.0)
    X_k = np.fft.fft2(x_k)

    historial = {
        'costo': [],
        'rmse': [],
        'cambio_relativo': []
    }

    for i in range(max_iter):

        # Gradiente en frecuencia
        residual_freq = H * X_k - Y

        grad_freq = (
            H_conj * residual_freq
            + lambda_reg * X_k
        )

        grad_espacial = np.real(
            np.fft.ifft2(grad_freq)
        )

        # Descenso de gradiente proyectado
        x_next = np.clip(
            x_k - tau * grad_espacial,
            0.0, 1.0
        )

        # Cambio relativo
        cambio_rel = (
            np.linalg.norm(x_next - x_k)
            / (np.linalg.norm(x_k) + 1e-12)
        )

        # Costo
        X_next = np.fft.fft2(x_next)

        degradada = np.real(
            np.fft.ifft2(H * X_next)
        )

        costo_fidelidad = (
            0.5 * np.sum((degradada - y)**2)
        )

        costo_reg = (
            0.5 * lambda_reg * np.sum(x_next**2)
        )

        historial['costo'].append(
            costo_fidelidad + costo_reg
        )

        historial['cambio_relativo'].append(cambio_rel)

        if x_true is not None:
            historial['rmse'].append(
                calcular_rmse(x_true, x_next)
            )

        x_k = x_next
        X_k = X_next

        if cambio_rel < tol:
            break

    return x_k, historial

def calcular_rmse(x_true, x_pred):
    return np.sqrt(np.mean((x_true.astype(np.float64) - x_pred.astype(np.float64))**2))