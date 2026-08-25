"""
Deney 4: Türev Hassasiyeti ve Hızı (Autodiff vs Sonlu Farklar)
=================================================================
Çok değişkenli fonksiyon:  f(x0, x1, x2) = x0^2 * x1 + sin(x2) * x0 + x1 * x2^2

Bu fonksiyonun gradyanı (1. türev) ve Hessian matrisi (2. türev) iki yöntemle
hesaplanır:

  1) NumPy ile Sonlu Farklar (Finite Differences)
     -> Türevin TANIMINA yakın bir yaklaşıklama:  f'(x) ~ (f(x+h) - f(x-h)) / 2h
     -> Adım büyüklüğü `h`'a bağlı yuvarlama/kesme hataları içerir.
     -> Hessian için "türevin türevini" almak gerektiğinden O(n^2) fonksiyon
        çağrısı gerekir; yavaştır.

  2) JAX ile Otomatik Türev (Automatic Differentiation)
     -> `jax.grad`      : gradyanı MAKİNE HASSASİYETİNDE, analitik olarak hesaplar.
     -> `jacfwd(jacrev(f))`: ileri-geri mod kombinasyonuyla Hessian'ı verimli
        şekilde hesaplar (Hessian hesaplamak için önerilen standart JAX deseni).

Otomatik türev, sayısal (finite difference) türevden FARKLI bir şeydir:
Yaklaşıklama yapmaz, zincir kuralını (chain rule) kodun kendisi üzerinde
otomatik olarak uygular ve tam (exact) sonuç üretir.

GPU'da iki ayrı süre raporlanır (compute-only vs end-to-end); nedeni ve
kapsamı için Deney 2'deki (02_kernel_fusion.py) açıklamaya bakınız. Burada
girdi tek bir 3 elemanlı vektör olduğundan transfer maliyeti ihmal edilebilir
düzeydedir; yine de metodoloji tutarlılığı için aynı ayrım uygulanır.

Not (adil kıyaslama): `jax_enable_x64` açık olduğundan NumPy ve JAX tarafı
AYNI hassasiyette (float64) çalışır; bu yüzden farklı bir dtype düzeltmesi
gerekmez.

Her yöntem (gradyan + Hessian birlikte, tek bir ölçüm birimi olarak) `timeit`
ile 10 kez ölçülür; ortalama ve standart sapma raporlanır. JIT derleme süresi
warmup çağrılarıyla ölçüm dışı bırakılır.
"""

import statistics
import timeit

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # hassas kıyaslama için float64


X0 = jnp.array([1.5, -2.0, 0.7])  # (x0, x1, x2) değerlendirme noktası
REPEATS = 10


# ---------------------------------------------------------------------------
# Ortak fonksiyon: hem NumPy hem JAX ile çağrılabilir (aynı matematiksel ifade)
# ---------------------------------------------------------------------------
def f(x):
    x0, x1, x2 = x[0], x[1], x[2]
    return x0**2 * x1 + jnp.sin(x2) * x0 + x1 * x2**2


def f_numpy(x: np.ndarray) -> float:
    x0, x1, x2 = x[0], x[1], x[2]
    return x0**2 * x1 + np.sin(x2) * x0 + x1 * x2**2


# ---------------------------------------------------------------------------
# 1) NumPy ile Sonlu Farklar
# ---------------------------------------------------------------------------
def numerical_gradient(x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f_numpy(x_plus) - f_numpy(x_minus)) / (2 * h)
    return grad


def numerical_hessian(x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_pp, x_pm, x_mp, x_mm = x.copy(), x.copy(), x.copy(), x.copy()
            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h
            hess[i, j] = (f_numpy(x_pp) - f_numpy(x_pm) - f_numpy(x_mp) + f_numpy(x_mm)) / (4 * h * h)
    return hess


# ---------------------------------------------------------------------------
# 2) JAX ile Otomatik Türev
# ---------------------------------------------------------------------------
jax_grad_fn = jax.jit(jax.grad(f))
jax_hessian_fn = jax.jit(jax.jacfwd(jax.jacrev(f)))  # Hessian için standart JAX deseni


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
    print(f"  {label:<28}: {mean_ms:6.3f} ± {std_ms:5.3f} ms (n={len(times)})")


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 4: Autodiff vs Sonlu Farklar (Gradyan & Hessian)")
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    x_np = np.array(X0, dtype=np.float64)

    # --- Referans: NumPy sonlu farklar (cihazdan bağımsız) ---
    numerical_times, (grad_numerical, hess_numerical) = time_it(
        lambda: (numerical_gradient(x_np), numerical_hessian(x_np))
    )
    numerical_mean = statistics.mean(numerical_times)

    print("\nGradyan (df/dx0, df/dx1, df/dx2):")
    print(f"  Sonlu Farklar : {grad_numerical}")

    print("\nHessian matrisi (Sonlu Farklar):")
    print("   ", str(hess_numerical).replace("\n", "\n    "))

    print()
    print_stats("NumPy Sonlu Farklar", numerical_times)

    cpu_mean = None
    gpu_compute_mean = None
    gpu_e2e_mean = None

    for name, device in devices.items():
        print(f"\n  --- {name} ---")

        if name != "GPU":
            # Tek cihaz ölçümü (CPU): girdi bir kez cihaza taşınır, sonrasında
            # SADECE derlenmiş fonksiyonların çalışma süresi ölçülür.
            x0_dev = jax.device_put(X0, device)

            # İlk çağrı: her iki fonksiyonun da JIT derlemesi + ilk çalıştırması.
            # Warmup görevini de görür, ama artık atılmıyor — süresi ayrıca
            # raporlanıyor (bkz. Deney 2'deki aynı ayrım).
            first_call_time = timeit.timeit(
                lambda: (jax_grad_fn(x0_dev).block_until_ready(), jax_hessian_fn(x0_dev).block_until_ready()),
                number=1,
            )
            print(f"  {'JAX first call (compile+run)':<28}: {first_call_time*1000:6.2f} ms")

            cpu_times, (grad_jax, hess_jax) = time_it(
                lambda: (jax_grad_fn(x0_dev).block_until_ready(), jax_hessian_fn(x0_dev).block_until_ready())
            )
            cpu_mean = statistics.mean(cpu_times)

            grad_diff = float(np.max(np.abs(grad_numerical - np.asarray(grad_jax))))
            hess_diff = float(np.max(np.abs(hess_numerical - np.asarray(hess_jax))))

            print(f"  jax.grad      : {np.asarray(grad_jax)}  (fark: {grad_diff:.2e})")
            print("  jacfwd(jacrev(f)) :")
            print("   ", str(np.asarray(hess_jax)).replace("\n", "\n    "))
            print(f"  Hessian maks. fark : {hess_diff:.2e}")
            print_stats(f"JAX {name} steady-state", cpu_times)
            continue

        # =====================================================================
        # GPU: iki ayrı, birbirine KARIŞTIRILMAMASI gereken ölçüm.
        # =====================================================================

        # --- Compute-only: girdi ÖNCEDEN GPU'ya taşınmış, warmup yapılmış.
        # Host<->GPU transferi ölçüme dahil DEĞİL. ---
        x0_gpu = jax.device_put(X0, device)

        jax_grad_fn(x0_gpu).block_until_ready()  # warmup (JIT derlemesi)
        jax_hessian_fn(x0_gpu).block_until_ready()  # warmup (JIT derlemesi)

        gpu_compute_times, (grad_jax, hess_jax) = time_it(
            lambda: (jax_grad_fn(x0_gpu).block_until_ready(), jax_hessian_fn(x0_gpu).block_until_ready())
        )
        gpu_compute_mean = statistics.mean(gpu_compute_times)

        grad_diff = float(np.max(np.abs(grad_numerical - np.asarray(grad_jax))))
        hess_diff = float(np.max(np.abs(hess_numerical - np.asarray(hess_jax))))

        print(f"  jax.grad      : {np.asarray(grad_jax)}  (fark: {grad_diff:.2e})")
        print("  jacfwd(jacrev(f)) :")
        print("   ", str(np.asarray(hess_jax)).replace("\n", "\n    "))
        print(f"  Hessian maks. fark : {hess_diff:.2e}")
        print_stats("JAX GPU compute-only", gpu_compute_times)

        # --- End-to-end (round-trip): host -> GPU -> hesapla -> host.
        # jax.device_get() hem hesaplamanın bitmesini bekler hem sonucu gerçekten
        # host belleğine kopyalar. Fonksiyon zaten derlenmiş; JIT bu ölçüme girmez. ---
        def round_trip():
            x0_gpu_iter = jax.device_put(x_np, device)
            grad_iter = jax_grad_fn(x0_gpu_iter)
            hess_iter = jax_hessian_fn(x0_gpu_iter)
            return jax.device_get(grad_iter), jax.device_get(hess_iter)

        round_trip()  # warmup: allocator/transfer yollarını ısıt (JIT zaten hazır)
        gpu_e2e_times, _ = time_it(round_trip)
        gpu_e2e_mean = statistics.mean(gpu_e2e_times)
        print_stats("JAX GPU end-to-end", gpu_e2e_times)

    print("\nHızlanma (NumPy'a göre, ortalama süreler üzerinden):")
    if cpu_mean is not None:
        print(f"  {'JAX CPU speedup':<28}: {numerical_mean / cpu_mean:.1f}x")
    if gpu_compute_mean is not None:
        print(f"  {'GPU compute-only speedup':<28}: {numerical_mean / gpu_compute_mean:.1f}x")
    if gpu_e2e_mean is not None:
        print(f"  {'GPU end-to-end speedup':<28}: {numerical_mean / gpu_e2e_mean:.1f}x")

    print("=" * 62)
