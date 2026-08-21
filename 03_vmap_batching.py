"""
Deney 3: Otomatik Vektörleştirme (vmap)
==========================================
Tek bir 2D vektör çifti için yazılmış Öklid mesafesi fonksiyonu, 50.000
nokta çifti üzerinde üç farklı yöntemle "toplu" (batch) hâle getirilir:

  1) Saf Python for döngüsü   -> fonksiyonu tek tek, 50.000 kez çağırır.
  2) NumPy eksen manipülasyonu -> fonksiyonu elle "vektörleştiririz"
                                   (kodu değiştirip axis=1 ekleriz).
  3) jax.vmap                  -> TEK vektör için yazılmış fonksiyonu
                                   hiç değiştirmeden otomatik olarak
                                   toplu çalışacak hâle getirir.

`vmap`'in gücü tam olarak budur: "tek örnek" mantığıyla düşünüp kodu
yazarsın, toplu (batched) hâle getirme işini JAX'e bırakırsın.
"""

import math
import time

import numpy as np

import jax
import jax.numpy as jnp


N_POINTS = 50_000


# ---------------------------------------------------------------------------
# Tek bir vektör çifti için yazılmış öklid mesafesi fonksiyonu
# (Bu fonksiyon SADECE tek bir çift için düşünülerek yazıldı.)
# ---------------------------------------------------------------------------
def euclidean_distance_single(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum((x - y) ** 2))


# ---------------------------------------------------------------------------
# 1) Saf Python for döngüsü
# ---------------------------------------------------------------------------
def python_loop_distance(X: list, Y: list) -> list:
    results = []
    for (x0, x1), (y0, y1) in zip(X, Y):
        d = math.sqrt((x0 - y0) ** 2 + (x1 - y1) ** 2)
        results.append(d)
    return results


# ---------------------------------------------------------------------------
# 2) NumPy: eksen manipülasyonu ile elle vektörleştirme
# ---------------------------------------------------------------------------
def numpy_vectorized_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # axis=1: her satır (nokta) için ayrı toplama
    return np.sqrt(np.sum((X - Y) ** 2, axis=1))


# ---------------------------------------------------------------------------
# 3) jax.vmap: tek-örnek fonksiyonunu otomatik olarak toplu hâle getirir
# ---------------------------------------------------------------------------
batched_distance = jax.jit(jax.vmap(euclidean_distance_single))


# ---------------------------------------------------------------------------
# Yardımcı: şık çıktı basma
# ---------------------------------------------------------------------------
def print_row(label: str, seconds: float) -> None:
    print(f"  {label:<32} {seconds*1000:>10.2f} ms")


if __name__ == "__main__":
    print("=" * 62)
    print(f" DENEY 3: vmap ile Otomatik Vektörleştirme ({N_POINTS:,} nokta)".replace(",", "."))
    print("=" * 62)

    rng = np.random.default_rng(0)
    X_np = rng.random((N_POINTS, 2))
    Y_np = rng.random((N_POINTS, 2))

    # --- 1) Saf Python (liste hâlinde) ---
    X_list = X_np.tolist()
    Y_list = Y_np.tolist()

    t0 = time.perf_counter()
    py_result = python_loop_distance(X_list, Y_list)
    t1 = time.perf_counter()
    py_time = t1 - t0

    # --- 2) NumPy vektörleştirme ---
    t0 = time.perf_counter()
    np_result = numpy_vectorized_distance(X_np, Y_np)
    t1 = time.perf_counter()
    np_time = t1 - t0

    # --- 3) jax.vmap: warmup (derleme) + gerçek ölçüm ---
    X_jax = jnp.asarray(X_np)
    Y_jax = jnp.asarray(Y_np)

    _ = batched_distance(X_jax, Y_jax).block_until_ready()  # warmup

    t0 = time.perf_counter()
    vmap_result = batched_distance(X_jax, Y_jax).block_until_ready()
    t1 = time.perf_counter()
    vmap_time = t1 - t0

    print("\nSonuçlar (saf çalışma süresi):\n")
    print_row("Python for döngüsü", py_time)
    print_row("NumPy (eksen manipülasyonu)", np_time)
    print_row("jax.vmap", vmap_time)

    print("\nHızlanma (Python'a göre):")
    print(f"  NumPy     : {py_time / np_time:>8.1f}x daha hızlı")
    print(f"  jax.vmap  : {py_time / vmap_time:>8.1f}x daha hızlı")

    max_diff = float(np.max(np.abs(np.asarray(vmap_result) - np_result)))
    print(f"\n  Maks. fark (NumPy vs vmap) : {max_diff:.2e}  (sayısal doğruluk kontrolü)")
    print("=" * 62)
