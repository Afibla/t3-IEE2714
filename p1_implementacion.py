import numpy as np
import matplotlib.pyplot as plt

# Aplicación padding con tamaño objetivo y tipo configurable
# tipos de padding: 'reflect', 'zeros' (o 'constant'), 'edge', 'wrap' con np.pad
def padding_imagen(img, size=None, tipo="reflect"):
    M, N = img.shape
    if size is None or tuple(size) == (M, N):
        return img.copy(), ((0,0),(0,0))
    P, Q = map(int, size)
    if P < M or Q < N:
        raise ValueError("El tamaño padded no puede ser menor que la imagen.")
    
    top = (P-M)//2
    left = (Q-N)//2
    bottom = P - M - top
    right =  Q - N - left

    mode = {"zeros":"constant", "constant":"constant"}.get(tipo, tipo)
    if mode not in ("constant","reflect","edge","wrap"):
        raise ValueError("Padding: reflect, zeros, edge o wrap.")
    kwargs = {"constant_values":0} if mode == "constant" else {}
    img_padded = np.pad(img, ((top,bottom),(left,right)), mode=mode, **kwargs)

    return img_padded.astype(np.float32, copy=False), ((top,bottom),(left,right))

# Recuperar imagen filtrada resultante de la IFFT 2D al tamaño original de la imagen
def recortar_imagen(imagen_padded, orig_shape, pad_width):
    M, N = orig_shape

    (pad_top, _), (pad_left, _) = pad_width

    return imagen_padded[
        pad_top:pad_top + M,
        pad_left:pad_left + N
    ]

# Construye la función de transferencia H(u,v) de un filtro Notch tipo = ['ideal', 'gaussiano']
def construir_filtro_notch(
    shape,
    notches,
    tipo='gaussiano',
    escala_u=1.0,
    escala_v=1.0
):
    P, Q = shape

    if tipo not in ['ideal', 'gaussiano']:
        raise ValueError("Tipo debe ser 'ideal' o 'gaussiano'.")

    # Crear grilla de coordenadas centradas en (P/2, Q/2)
    u = np.arange(P, dtype=np.float32) - P // 2
    v = np.arange(Q, dtype=np.float32) - Q // 2
    U, V = np.meshgrid(u, v, indexing="ij")

    H = np.ones((P, Q), dtype=np.float32)

    for uk, vk, D0 in notches:
        if D0 <= 0:
            raise ValueError("D0 debe ser positivo.")

        # Centros escalados al tamaño de la imagen con padding
        uk_scaled = uk * escala_u
        vk_scaled = vk * escala_v

        # Distancias expresadas en las unidades originales
        du1 = (U - uk_scaled) / escala_u
        dv1 = (V - vk_scaled) / escala_v

        du2 = (U + uk_scaled) / escala_u
        dv2 = (V + vk_scaled) / escala_v

        D1 = du1**2 + dv1**2
        D2 = du2**2 + dv2**2

        if tipo == "ideal":
            H[(D1 <= D0**2) | (D2 <= D0**2)] = 0

        else:
            H *= (1 - np.exp(-D1 / (2 * D0**2))).astype(np.float32)
            H *= (1 - np.exp(-D2 / (2 * D0**2))).astype(np.float32)

    return H

def filtrar_frecuencia_notch(
        img, notches, tipo_notch="gaussiano",
        modo_padding="reflect", factor_padding=1.0,
        devolver_detalle=False):

    M, N = img.shape
    if factor_padding < 1:
        raise ValueError("factor_padding debe ser >= 1.")
    if factor_padding == 1 or modo_padding == "none":
        target, tipo_pad = (M,N), "reflect"
    else:
        target = (int(np.ceil(M*factor_padding)), int(np.ceil(N*factor_padding)))
        tipo_pad = modo_padding

    # 1. Padding
    img_pad, pad_width = padding_imagen(img, target, tipo_pad)
    P, Q = img_pad.shape

    # escalas de frecuencia
    su, sv = P/M, Q/N

    # 2. FFT y centrado
    F = np.fft.fftshift(np.fft.fft2(img_pad))

    # 3. Construcción del filtro
    H = construir_filtro_notch((P,Q), notches, tipo_notch, escala_u=su, escala_v=sv)

    # 4. Filtrado en frecuencia
    G = F * H

    G_unshift = np.fft.ifftshift(G)
    g_padded = np.real(
        np.fft.ifft2(G_unshift)
    )

    # Filtrada y diferencia
    g_filtrada = recortar_imagen(g_padded, (M, N), pad_width)
    diferencia = np.abs(img - g_filtrada)

    out = {"imagen":g_filtrada, "diferencia":diferencia,
           "pad_width":pad_width, "shape_padding":(P,Q),
           "escala_u":su, "escala_v":sv}

    if devolver_detalle:
        out.update({"mascara":H, "espectro_antes":np.log(1+np.abs(F)),
                    "espectro_despues":np.log(1+np.abs(G)),
                    "F":F, "G":G})
    return out