"""
Deney 4: Türev Hassasiyeti ve Hızı (Autodiff vs Sonlu Farklar)
=================================================================
Çok değişkenli fonksiyon:  f(x0, x1, x2) = x0^2 * x1 + sin(x2) * x0 + x1 * x2^2

Bu fonksiyonun gradyanı (1. türev) ve Hessian matrisi (2. türev) iki yöntemle
hesaplanır:

  1) NumPy ile Sonlu Farklar (Finite Differences)
     -> Türevin TANIMINA yakın bir yaklaşıklama:  f'(x) ~ (f(x+h) - f(x-h)) / 2h
     -> Bir `h` adım büyüklüğü seçmeyi gerektirir; sonuç hem kesme (truncation)
        hem yuvarlama (rounding) hatası içerir ve `h` seçimine duyarlıdır.
     -> Dense Hessian için her (i, j) çifti ayrı ayrı örneklendiğinden, gereken
        fonksiyon değerlendirmesi sayısı girdi boyutu n büyüdükçe yaklaşık
        O(n^2) mertebesinde artar.

  2) JAX ile Otomatik Türev (Automatic Differentiation)
     -> `jax.grad`: sonlu fark yaklaşıklaması KULLANMAZ. Programın primitive
        işlemleri üzerinde zincir kuralını (chain rule) otomatik uygulayarak
        türev fonksiyonunu türetir.
     -> `jacfwd(jacrev(f))`: ileri-geri mod kombinasyonuyla Hessian'ı hesaplar
        (Hessian için önerilen standart JAX deseni).

Otomatik türev, sayısal (finite difference) türevden FARKLI bir şeydir. Yine de
sonuçlar floating-point aritmetiğiyle değerlendirildiği için gerçek sayılar
anlamında "tam exact" değildir; autodiff'in kazancı, bir `h` adım büyüklüğü
seçmeyi ve truncation-error yaklaşıklamasını gerektirmemesidir.

GPU'da iki ayrı süre raporlanır (compute-only vs end-to-end); nedeni ve
kapsamı için Deney 2'deki (02_kernel_fusion.py) açıklamaya bakınız. Burada
girdi tek bir 3 elemanlı vektör olduğundan problem GPU için çok küçüktür:
kernel launch ve dispatch maliyetleri hesabın kendisine kıyasla baskındır, bu
yüzden GPU'nun bu deneyde CPU'ya karşı avantaj göstermemesi beklenen bir
sonuçtur. Workload bu nedenle büyütülmemiştir.

Not (adil kıyaslama): `jax_enable_x64` açık olduğundan NumPy ve JAX tarafı
AYNI hassasiyette (float64) çalışır; bu yüzden farklı bir dtype düzeltmesi
gerekmez.

Benchmark metodolojisi
----------------------
Ölçülen süreler mikrosaniye mertebesinde olduğundan, `timeit.repeat(number=1)`
timer çözünürlüğü ve scheduler gürültüsü yüzünden yüksek varyans üretir. Bunun
yerine her `timeit` tekrarında fonksiyon `number` kez çalıştırılır ve toplam
süre `number`'a bölünür. `number` sabit değildir: her ölçümden önce tek bir
sıcak (warm) çağrı ölçülüp her tekrarın ~200 ms sürmesini hedefleyen bir değer
seçilir (kalibrasyon çağrısı sonuçlara dahil edilmez). Bu, YAPILAN İŞİ
DEĞİŞTİRMEZ — aynı gradyan + Hessian hesabı `number` kez tekrarlanır ve
raporlanan süre daima TEK bir gradyan + Hessian hesabı başınadır.

JAX'in ilk çağrısı (tracing + autodiff dönüşümü + JIT derlemesi + çalıştırma)
`number=1` ile ayrıca ölçülür ve steady-state ölçümüne karıştırılmaz; sonraki
ölçümler cache'deki derlenmiş executable'ın çalışma süresini gösterir. Her
timed JAX çağrısı `.block_until_ready()` ile senkronize edilir (JAX asenkron
dispatch yapar; timer hesap gerçekten bitmeden durmamalıdır).
"""

import math
import statistics
import timeit

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # hassas kıyaslama için float64


X0 = jnp.array([1.5, -2.0, 0.7])  # (x0, x1, x2) değerlendirme noktası
REPEATS = 10

# Her timeit tekrarının hedeflenen gerçek ölçüm penceresi (~100-300 ms bandı).
TARGET_SECONDS = 0.2
MIN_INNER_LOOPS = 1
MAX_INNER_LOOPS = 100_000


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


def numpy_derivatives(x: np.ndarray):
    """Tek bir benchmark birimi: gradyan + Hessian birlikte."""
    return numerical_gradient(x), numerical_hessian(x)


# ---------------------------------------------------------------------------
# 2) JAX ile Otomatik Türev
# ---------------------------------------------------------------------------
jax_grad_fn = jax.jit(jax.grad(f))
jax_hessian_fn = jax.jit(jax.jacfwd(jax.jacrev(f)))  # Hessian için standart JAX deseni


def run_jax_derivatives(x):
    """Tek bir benchmark birimi: gradyan + Hessian, ikisi de senkronize edilmiş.

    JAX asenkron dispatch yaptığı için timer'ın erken durmaması adına HER İKİ
    sonuç da `.block_until_ready()` ile beklenir.
    """
    grad = jax_grad_fn(x)
    hess = jax_hessian_fn(x)
    grad.block_until_ready()
    hess.block_until_ready()
    return grad, hess


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
# Benchmark yardımcıları (diğer deneylere taşınabilir olacak şekilde ayrıldı)
# ---------------------------------------------------------------------------
def choose_inner_loops(
    single_call_seconds: float,
    target_seconds: float = TARGET_SECONDS,
    min_number: int = MIN_INNER_LOOPS,
    max_number: int = MAX_INNER_LOOPS,
) -> int:
    """Her timeit tekrarının ~`target_seconds` sürmesi için gereken iç döngü sayısı.

    Bu sayı YAPILAN İŞİ değiştirmez; aynı çağrı `number` kez tekrarlanır ve
    toplam süre `number`'a bölünerek çağrı başına süreye normalize edilir.
    Amaç, ölçüm penceresini timer/scheduler gürültüsünün yanında büyük tutmaktır.
    """
    if single_call_seconds <= 0:
        return max_number
    number = round(target_seconds / single_call_seconds)
    return int(max(min_number, min(number, max_number)))


def calibrate_inner_loops(
    fn,
    target_seconds: float = TARGET_SECONDS,
    min_number: int = MIN_INNER_LOOPS,
    max_number: int = MAX_INNER_LOOPS,
) -> int:
    """fn ZATEN sıcakken (warmup / first call sonrası) tek bir çağrıyı ölçüp
    uygun iç döngü sayısını seçer. Bu kalibrasyon çağrısı sonuçlara girmez."""
    single = timeit.timeit(fn, number=1)
    return choose_inner_loops(single, target_seconds, min_number, max_number)


def benchmark_steady(
    fn,
    repeats: int = REPEATS,
    target_seconds: float = TARGET_SECONDS,
    min_number: int = MIN_INNER_LOOPS,
    max_number: int = MAX_INNER_LOOPS,
) -> dict:
    """Kalibre edilmiş iç döngüyle steady-state ölçüm.

    fn'in ÇAĞRILMADAN ÖNCE sıcak olması beklenir (Python için untimed warmup,
    JAX için ayrıca ölçülen first call). Dönen süreler ÇAĞRI BAŞINA'dır.
    """
    result = None

    def call():
        nonlocal result
        result = fn()

    number = calibrate_inner_loops(call, target_seconds, min_number, max_number)
    totals = timeit.repeat(call, number=number, repeat=repeats)
    per_call_times = [total / number for total in totals]

    return {
        "times": per_call_times,
        "number": number,
        "mean": statistics.mean(per_call_times),
        "std": statistics.stdev(per_call_times) if len(per_call_times) > 1 else 0.0,
        "result": result,
    }


def fmt_ms(seconds: float) -> str:
    ms = seconds * 1000
    return f"{ms:.4f} ms" if ms < 1 else f"{ms:.3f} ms"


def print_stats(label: str, stats: dict) -> None:
    mean_ms = stats["mean"] * 1000
    std_ms = stats["std"] * 1000
    decimals = 4 if mean_ms < 1 else 3
    print(
        f"    {label:<22}: {mean_ms:.{decimals}f} ± {std_ms:.{decimals}f} ms "
        f"(repeats={len(stats['times'])}, inner loops={stats['number']})"
    )


def relative_performance(baseline: float, candidate: float) -> str:
    if candidate <= baseline:
        return f"{baseline / candidate:.1f}x daha hızlı"
    return f"{candidate / baseline:.1f}x daha yavaş"


def break_even_repeats(first_call_time: float, steady_mean: float, baseline_mean: float):
    """Derleme maliyeti kaç tekrarda amorti olur?

      first_call + (n-1)*steady <= n*baseline
      =>  n >= (first_call - steady) / (baseline - steady)

    steady >= baseline ise hiçbir zaman amorti olmaz (None döner).
    """
    if baseline_mean <= steady_mean:
        return None
    n = (first_call_time - steady_mean) / (baseline_mean - steady_mean)
    return max(1, math.ceil(n))


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 4: Autodiff vs Sonlu Farklar (Gradyan & Hessian)")
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    print("\nBenchmark ayarları:")
    print(f"  repeats               = {REPEATS}")
    print(f"  hedef ölçüm penceresi ~= {TARGET_SECONDS * 1000:.0f} ms / repeat")
    print("  raporlanan süre       = tek bir (gradyan + Hessian) hesabı başına")

    x_np = np.array(X0, dtype=np.float64)

    # -----------------------------------------------------------------------
    # Referans: NumPy sonlu farklar (cihazdan bağımsız)
    # -----------------------------------------------------------------------
    numpy_derivatives(x_np)  # untimed warmup (NumPy'da ölçülecek derleme maliyeti yok)
    numpy_stats = benchmark_steady(lambda: numpy_derivatives(x_np))
    grad_numerical, hess_numerical = numpy_stats["result"]
    numerical_mean = numpy_stats["mean"]

    print("\nNumPy sonlu farklar:")
    print_stats("steady-state", numpy_stats)

    print("\n  Gradyan (df/dx0, df/dx1, df/dx2):")
    print(f"    {grad_numerical}")
    print("  Hessian matrisi:")
    print("     ", str(hess_numerical).replace("\n", "\n      "))

    cpu = None
    gpu_compute = None
    gpu_e2e = None
    inner_loops = {"NumPy sonlu farklar": numpy_stats["number"]}
    diffs = {}

    for name, device in devices.items():
        print(f"\n  --- {name} ---")

        if name != "GPU":
            # Tek cihaz ölçümü (CPU): girdi bir kez cihaza taşınır, sonrasında
            # SADECE derlenmiş fonksiyonların çalışma süresi ölçülür.
            x0_dev = jax.device_put(X0, device)

            # İlk çağrı: tracing + autodiff dönüşümü + JIT derlemesi + ilk
            # çalıştırma. number=1 ile ölçülür; iç döngü UYGULANMAZ ve
            # steady-state kalibrasyonuna karıştırılmaz. Aynı zamanda warmup
            # görevi görür.
            cpu_first_call = timeit.timeit(lambda: run_jax_derivatives(x0_dev), number=1)
            print(f"    {'first call (compile+run)':<22}: {fmt_ms(cpu_first_call)}")

            cpu = benchmark_steady(lambda: run_jax_derivatives(x0_dev))
            cpu["first_call"] = cpu_first_call
            inner_loops["JAX CPU"] = cpu["number"]
            print_stats("steady-state", cpu)

            grad_jax, hess_jax = cpu["result"]
            grad_diff = float(np.max(np.abs(grad_numerical - np.asarray(grad_jax))))
            hess_diff = float(np.max(np.abs(hess_numerical - np.asarray(hess_jax))))
            diffs[name] = (grad_diff, hess_diff)

            print(f"    jax.grad            : {np.asarray(grad_jax)}")
            print("    jacfwd(jacrev(f))   :")
            print("       ", str(np.asarray(hess_jax)).replace("\n", "\n        "))
            print(f"    gradyan maks. fark  : {grad_diff:.2e}")
            print(f"    Hessian maks. fark  : {hess_diff:.2e}")
            continue

        # =====================================================================
        # GPU: iki ayrı, birbirine KARIŞTIRILMAMASI gereken ölçüm.
        # =====================================================================

        # --- Compute-only: girdi ÖNCEDEN GPU'ya taşınır ve iç döngü boyunca
        # ORADA KALIR. Host<->GPU transferi ölçüme dahil DEĞİL. ---
        x0_gpu = jax.device_put(X0, device)

        # CPU ile metodolojik tutarlılık: GPU'nun ilk çağrısı da ayrıca ölçülür
        # (GPU backend'i için ayrı bir derleme yapılır). Bu süre speedup
        # hesabına KATILMAZ; yalnız cold-start bilgisidir.
        gpu_first_call = timeit.timeit(lambda: run_jax_derivatives(x0_gpu), number=1)
        print(f"    {'first call (compile+run)':<22}: {fmt_ms(gpu_first_call)}")

        gpu_compute = benchmark_steady(lambda: run_jax_derivatives(x0_gpu))
        gpu_compute["first_call"] = gpu_first_call
        inner_loops["JAX GPU compute-only"] = gpu_compute["number"]
        print_stats("compute-only", gpu_compute)

        grad_jax, hess_jax = gpu_compute["result"]
        grad_diff = float(np.max(np.abs(grad_numerical - np.asarray(grad_jax))))
        hess_diff = float(np.max(np.abs(hess_numerical - np.asarray(hess_jax))))
        diffs[name] = (grad_diff, hess_diff)

        # --- End-to-end (round-trip): host -> GPU -> hesapla -> host.
        # Her iç döngü çağrısı GERÇEK bir round-trip içerir (device_put ölçümün
        # içinde kalır; dışarı alınırsa bu ölçüm compute-only'ye dönüşürdü).
        # jax.device_get() hem hesabın bitmesini bekler hem sonucu host belleğine
        # kopyalar; bu yüzden ayrıca block_until_ready() eklenmez.
        # Fonksiyon zaten derlenmiş; JIT bu ölçüme girmez. ---
        def round_trip():
            x0_gpu_iter = jax.device_put(x_np, device)
            grad_iter = jax_grad_fn(x0_gpu_iter)
            hess_iter = jax_hessian_fn(x0_gpu_iter)
            return jax.device_get(grad_iter), jax.device_get(hess_iter)

        round_trip()  # untimed warmup: allocator/transfer yollarını ısıt (JIT zaten hazır)
        gpu_e2e = benchmark_steady(round_trip)
        inner_loops["JAX GPU end-to-end"] = gpu_e2e["number"]
        print_stats("end-to-end", gpu_e2e)

        print(f"    jax.grad            : {np.asarray(grad_jax)}")
        print("    jacfwd(jacrev(f))   :")
        print("       ", str(np.asarray(hess_jax)).replace("\n", "\n        "))
        print(f"    gradyan maks. fark  : {grad_diff:.2e}")
        print(f"    Hessian maks. fark  : {hess_diff:.2e}")

    # -----------------------------------------------------------------------
    # Özet
    # -----------------------------------------------------------------------
    print("\nKullanılan iç döngü (inner loop) sayıları — kalibrasyonla seçildi:")
    for label, number in inner_loops.items():
        print(f"  {label:<24}: {number}")

    print("\nGöreli performans (NumPy sonlu farklara göre, steady-state ortalamaları):")
    if cpu is not None:
        print(f"  {'JAX CPU':<24}: {relative_performance(numerical_mean, cpu['mean'])}")
    if gpu_compute is not None:
        print(f"  {'GPU compute-only':<24}: {relative_performance(numerical_mean, gpu_compute['mean'])}")
    if gpu_e2e is not None:
        print(f"  {'GPU end-to-end':<24}: {relative_performance(numerical_mean, gpu_e2e['mean'])}")

    print("\nBreak-even (ilk çağrının derleme maliyeti kaç tekrarda amorti oluyor):")

    def be_str(first_call, steady_mean):
        n = break_even_repeats(first_call, steady_mean, numerical_mean)
        return f"~{n} tekrar" if n is not None else "asla (steady-state NumPy'dan yavaş)"

    if cpu is not None:
        print(f"  {'JAX CPU':<24}: {be_str(cpu['first_call'], cpu['mean'])}")
    if gpu_compute is not None:
        print(f"  {'GPU compute-only':<24}: {be_str(gpu_compute['first_call'], gpu_compute['mean'])}")
    if gpu_e2e is not None:
        print(f"  {'GPU end-to-end':<24}: {be_str(gpu_compute['first_call'], gpu_e2e['mean'])}")

    print("\nSayısal kontrol (maks. mutlak fark, sonlu farklar referans alınarak):")
    for name, (grad_diff, hess_diff) in diffs.items():
        print(f"  {name:<24}: gradyan {grad_diff:.2e} | Hessian {hess_diff:.2e}")

    print("=" * 62)
