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

GPU'da iki ayrı süre raporlanır (compute-only vs end-to-end); nedeni ve
kapsamı için Deney 2'deki (02_kernel_fusion.py) açıklamaya bakınız.

Not (adil kıyaslama): NumPy dizileri BİLEREK float32 olarak oluşturuluyor.
JAX varsayılan olarak float64'ü float32'ye sessizce indirger (x64 modu kapalı);
NumPy tarafı float64 bırakılsaydı iki taraf farklı hassasiyette çalışır ve
hız kıyaslaması yanıltıcı olurdu.

Her yöntem `timeit` ile 10 kez ölçülür; ortalama ve standart sapma raporlanır.
JIT derleme süresi warmup çağrılarıyla ölçüm dışı bırakılır; veri oluşturma
(rastgele dizi üretimi) hiçbir ölçüme dahil değildir.
"""

import math
import statistics
import timeit

import numpy as np

import jax
import jax.numpy as jnp


N_POINTS = 50_000
REPEATS = 10


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
# fn'i timeit ile `repeat` kez ölçer; (süreler [s], son çağrının sonucu) döndürür.
# ---------------------------------------------------------------------------
def time_it(fn, repeat: int = REPEATS):
    result = None

    def call():
        nonlocal result
        result = fn()

    times = timeit.repeat(call, number=1, repeat=repeat)
    return times, result


def print_stats(label: str, times: list) -> None:
    mean_ms = statistics.mean(times) * 1000
    std_ms = statistics.stdev(times) * 1000
    print(f"  {label:<28}: {mean_ms:6.2f} ± {std_ms:5.2f} ms (n={len(times)})")


if __name__ == "__main__":
    print("=" * 62)
    print(f" DENEY 3: vmap ile Otomatik Vektörleştirme ({N_POINTS:,} nokta)".replace(",", "."))
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    # Veri BİR KEZ, host (NumPy) belleğinde üretilir; üretim süresi hiçbir
    # ölçüme dahil edilmez. float32: JAX'in varsayılan hassasiyetiyle eşleşsin
    # diye bilinçli olarak seçildi (bkz. dosya başındaki not).
    rng = np.random.default_rng(0)
    X_np = rng.random((N_POINTS, 2)).astype(np.float32)
    Y_np = rng.random((N_POINTS, 2)).astype(np.float32)

    # --- 1) Saf Python (liste hâlinde, cihazdan bağımsız) ---
    X_list = X_np.tolist()
    Y_list = Y_np.tolist()

    py_times, py_result = time_it(lambda: python_loop_distance(X_list, Y_list))
    py_mean = statistics.mean(py_times)

    # --- 2) NumPy vektörleştirme (cihazdan bağımsız) ---
    np_times, np_result = time_it(lambda: numpy_vectorized_distance(X_np, Y_np))
    np_mean = statistics.mean(np_times)

    print("\nSonuçlar (ortalama ± std, n=10):\n")
    print_stats("Python for döngüsü", py_times)
    print_stats("NumPy (eksen manipülasyonu)", np_times)

    print("\nHızlanma (Python'a göre):")
    print(f"  {'NumPy':<28}: {py_mean / np_mean:.1f}x")

    cpu_mean = None
    gpu_compute_mean = None
    gpu_e2e_mean = None

    # --- 3) Her bulunan cihazda (CPU, varsa GPU) jax.vmap çalıştır ---
    for name, device in devices.items():
        print(f"\n  --- {name} ---")

        if name != "GPU":
            # Tek cihaz ölçümü (CPU): veri bir kez cihaza taşınır, sonrasında
            # SADECE derlenmiş fonksiyonun çalışma süresi ölçülür.
            X_dev = jax.device_put(X_np, device)
            Y_dev = jax.device_put(Y_np, device)

            # İlk çağrı: JIT derlemesi + ilk çalıştırmayı birlikte içerir. Bu
            # çağrı warmup görevini de görür, ama artık atılmıyor — süresi
            # ayrıca raporlanıyor (bkz. Deney 2'deki aynı ayrım).
            first_call_time = timeit.timeit(lambda: batched_distance(X_dev, Y_dev).block_until_ready(), number=1)
            print(f"  {'jax.vmap first call (compile+run)':<28}: {first_call_time*1000:6.2f} ms")

            cpu_times, result_dev = time_it(lambda: batched_distance(X_dev, Y_dev).block_until_ready())
            cpu_mean = statistics.mean(cpu_times)
            print_stats(f"jax.vmap steady-state [{name}]", cpu_times)

            max_diff = float(np.max(np.abs(np.asarray(result_dev) - np_result)))
            print(f"  {'Maks. fark (NumPy vs vmap)':<28}: {max_diff:.2e}")
            continue

        # =====================================================================
        # GPU: iki ayrı, birbirine KARIŞTIRILMAMASI gereken ölçüm.
        # =====================================================================

        # --- Compute-only: veri ÖNCEDEN GPU'ya taşınmış, warmup yapılmış.
        # Host<->GPU transferi ölçüme dahil DEĞİL. ---
        X_gpu = jax.device_put(X_np, device)
        Y_gpu = jax.device_put(Y_np, device)

        batched_distance(X_gpu, Y_gpu).block_until_ready()  # warmup (JIT derlemesi)

        gpu_compute_times, result_gpu = time_it(lambda: batched_distance(X_gpu, Y_gpu).block_until_ready())
        gpu_compute_mean = statistics.mean(gpu_compute_times)
        print_stats("jax.vmap GPU compute-only", gpu_compute_times)

        # --- End-to-end (round-trip): host -> GPU -> hesapla -> host.
        # jax.device_get() hem hesaplamanın bitmesini bekler hem sonucu gerçekten
        # host belleğine kopyalar. Fonksiyon zaten derlenmiş; JIT bu ölçüme girmez. ---
        def round_trip():
            X_gpu_iter = jax.device_put(X_np, device)
            Y_gpu_iter = jax.device_put(Y_np, device)
            result_iter = batched_distance(X_gpu_iter, Y_gpu_iter)
            return jax.device_get(result_iter)  # GPU -> host, NumPy dizisi olarak döner

        round_trip()  # warmup: allocator/transfer yollarını ısıt (JIT zaten hazır)
        gpu_e2e_times, result_gpu_host = time_it(round_trip)
        gpu_e2e_mean = statistics.mean(gpu_e2e_times)
        print_stats("jax.vmap GPU end-to-end", gpu_e2e_times)

        max_diff = float(np.max(np.abs(result_gpu_host - np_result)))
        print(f"  {'Maks. fark (NumPy vs vmap)':<28}: {max_diff:.2e}")

    print("\nHızlanma (Python'a göre, ortalama süreler üzerinden):")
    if cpu_mean is not None:
        print(f"  {'jax.vmap CPU speedup':<28}: {py_mean / cpu_mean:.1f}x")
    if gpu_compute_mean is not None:
        print(f"  {'GPU compute-only speedup':<28}: {py_mean / gpu_compute_mean:.1f}x")
    if gpu_e2e_mean is not None:
        print(f"  {'GPU end-to-end speedup':<28}: {py_mean / gpu_e2e_mean:.1f}x")

    print("=" * 62)
