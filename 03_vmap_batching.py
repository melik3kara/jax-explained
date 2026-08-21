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


# ---------------------------------------------------------------------------
# Yardımcı: bu makinede bulunan JAX cihazlarını (CPU / GPU) tespit et.
# GPU yoksa jax.devices("gpu") hata fırlatır; bunu sessizce yakalayıp atlıyoruz.
# ---------------------------------------------------------------------------
def get_available_devices() -> dict:
    devices = {}
    try:
        devices["CPU"] = jax.devices("cpu")[0]
    except RuntimeError:
        pass
    try:
        devices["GPU"] = jax.devices("gpu")[0]
    except RuntimeError:
        pass
    return devices


# ---------------------------------------------------------------------------
# Belirli bir cihazda jax.vmap'i çalıştırır (warmup + gerçek ölçüm).
# `jax.default_device` bloğu içinde oluşturulan diziler o cihaza yerleşir.
# ---------------------------------------------------------------------------
def run_on_device(device, X_np: np.ndarray, Y_np: np.ndarray):
    with jax.default_device(device):
        X_jax = jnp.asarray(X_np)
        Y_jax = jnp.asarray(Y_np)

        _ = batched_distance(X_jax, Y_jax).block_until_ready()  # warmup

        t0 = time.perf_counter()
        vmap_result = batched_distance(X_jax, Y_jax).block_until_ready()
        elapsed = time.perf_counter() - t0

    return elapsed, vmap_result


if __name__ == "__main__":
    print("=" * 62)
    print(f" DENEY 3: vmap ile Otomatik Vektörleştirme ({N_POINTS:,} nokta)".replace(",", "."))
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    rng = np.random.default_rng(0)
    X_np = rng.random((N_POINTS, 2))
    Y_np = rng.random((N_POINTS, 2))

    # --- 1) Saf Python (liste hâlinde, cihazdan bağımsız) ---
    X_list = X_np.tolist()
    Y_list = Y_np.tolist()

    t0 = time.perf_counter()
    py_result = python_loop_distance(X_list, Y_list)
    py_time = time.perf_counter() - t0

    # --- 2) NumPy vektörleştirme (cihazdan bağımsız) ---
    t0 = time.perf_counter()
    np_result = numpy_vectorized_distance(X_np, Y_np)
    np_time = time.perf_counter() - t0

    print("\nSonuçlar (saf çalışma süresi):\n")
    print_row("Python for döngüsü", py_time)
    print_row("NumPy (eksen manipülasyonu)", np_time)

    print("\nHızlanma (Python'a göre):")
    print(f"  NumPy     : {py_time / np_time:>8.1f}x daha hızlı")

    # --- 3) Her bulunan cihazda (CPU, varsa GPU) jax.vmap çalıştır ---
    for name, device in devices.items():
        vmap_time, vmap_result = run_on_device(device, X_np, Y_np)
        max_diff = float(np.max(np.abs(np.asarray(vmap_result) - np_result)))

        print(f"\n  --- {name} ---")
        print_row(f"jax.vmap [{name}]", vmap_time)
        print(f"  [{name}] jax.vmap  : {py_time / vmap_time:>8.1f}x daha hızlı (Python'a göre)")
        print(f"  Maks. fark (NumPy vs vmap) : {max_diff:.2e}  (sayısal doğruluk kontrolü)")

    print("=" * 62)
