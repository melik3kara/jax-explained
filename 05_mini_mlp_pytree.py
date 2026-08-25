"""
Deney 5: Pytree ile Saf Fonksiyonel Eğitim Adımı
===================================================
Hiçbir harici framework (Flax, Optax, PyTorch, ...) kullanmadan, sadece
`jax.grad` ve `@jax.jit` ile 2 katmanlı basit bir MLP'nin TEK bir eğitim
adımını (loss -> grad -> update) simüle ederiz.

Pytree nedir?
  JAX'te ağırlıklar tek bir dev vektör değil, iç içe dict/list/tuple
  yapıları (Pytree) içinde tutulabilir. `jax.grad`, bu yapının TAMAMI
  için otomatik olarak, aynı şekle (shape) sahip bir gradyan Pytree'si
  döndürür. Bu sayede ağırlıkları elle "düzleştirmeye" (flatten) gerek
  kalmadan doğal bir sözlük (dict) yapısıyla çalışabiliriz.

Model: x -> Linear(W1, b1) -> tanh -> Linear(W2, b2) -> y_pred

Deneyin amacı (önemli): Bu deneyin temel amacı Pytree + autodiff + functional
SGD akışını göstermektir; model çok küçük olduğu için GPU'nun CPU'dan hızlı
olması beklenen bir gereklilik DEĞİLDİR. Bu ölçekte kernel launch / dispatch
maliyetleri hesabın kendisine baskın gelir; raporlanan CPU/GPU karşılaştırması
yalnızca bağlam bilgisidir, deneyin başarı ölçütü değildir.

GPU'da iki ayrı süre raporlanır (compute-only vs end-to-end); nedeni ve
kapsamı için Deney 2'deki (02_kernel_fusion.py) açıklamaya bakınız. Burada
girdi/ağırlıklar küçük olduğundan transfer maliyeti ihmal edilebilir
düzeydedir; yine de metodoloji tutarlılığı için aynı ayrım uygulanır.

Sayısal not: CPU ve GPU'da hesaplanan kayıp değerleri arasında (ör. 4.699378
vs 4.699393 gibi) çok küçük farklar görülebilir. Bu, farklı backend'lerin
farklı kernel/işlem sıralaması kullanmasından kaynaklanan normal bir
floating-point farkıdır; tam eşitlik (exact equality) beklenmez.

Benchmark metodolojisi (Deney 4 ile aynı)
-----------------------------------------
Tek bir eğitim adımı mikrosaniye-milisaniye mertebesinde sürdüğü için
`timeit.repeat(number=1)` timer/scheduler gürültüsü yüzünden yüksek varyans
üretir. Bunun yerine her `timeit` tekrarında adım `number` kez çalıştırılır ve
toplam süre `number`'a bölünür. `number` sabit değildir: ölçümden önce tek bir
sıcak (warm) çağrı ölçülüp her tekrarın ~200 ms sürmesini hedefleyen bir değer
seçilir (kalibrasyon çağrısı sonuçlara dahil edilmez). Raporlanan süre daima
TEK BİR eğitim adımı başınadır.

Önemli: her iç döngü çağrısı AYNI başlangıç parametreleriyle çalışır
(`train_step(params_dev, X_dev, Y_dev)`); dönen `new_params` bir sonraki
çağrıya beslenmez. Ölçmek istediğimiz şey "tek bir eğitim adımının maliyeti",
ardışık bir eğitim döngüsü değildir.

JAX'in ilk çağrısı (tracing + JIT derlemesi + çalıştırma) `number=1` ile
ayrıca ölçülür ve steady-state ölçümüne karıştırılmaz. Her timed
compute-only çağrısı `jax.block_until_ready(...)` ile senkronize edilir (JAX
asenkron dispatch yapar; timer hesap gerçekten bitmeden durmamalıdır).
"""

import statistics
import timeit

import numpy as np

import jax
import jax.numpy as jnp


IN_DIM = 4
HIDDEN_DIM = 8
OUT_DIM = 1
N_SAMPLES = 256
LEARNING_RATE = 0.1
REPEATS = 10

# Her timeit tekrarının hedeflenen gerçek ölçüm penceresi (~100-300 ms bandı).
TARGET_SECONDS = 0.2
MIN_INNER_LOOPS = 1
MAX_INNER_LOOPS = 100_000


# ---------------------------------------------------------------------------
# Ağırlıkları bir Pytree (dict) içinde ilklendir
# ---------------------------------------------------------------------------
def init_params(key):
    k1, k2 = jax.random.split(key)
    return {
        "W1": jax.random.normal(k1, (IN_DIM, HIDDEN_DIM)) * 0.1,
        "b1": jnp.zeros(HIDDEN_DIM),
        "W2": jax.random.normal(k2, (HIDDEN_DIM, OUT_DIM)) * 0.1,
        "b2": jnp.zeros(OUT_DIM),
    }


# ---------------------------------------------------------------------------
# İleri yayılım (forward pass): 2 katmanlı basit MLP
# ---------------------------------------------------------------------------
def forward(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    y_pred = h @ params["W2"] + params["b2"]
    return y_pred


# ---------------------------------------------------------------------------
# Kayıp fonksiyonu: ortalama kare hata (MSE)
# ---------------------------------------------------------------------------
def loss_fn(params, x, y):
    y_pred = forward(params, x)
    return jnp.mean((y_pred - y) ** 2)


# ---------------------------------------------------------------------------
# Tek eğitim adımı: loss -> grad -> parametre güncelleme (elle SGD)
# `jax.grad`, params ile AYNI Pytree yapısında bir gradyan döndürür,
# bu yüzden ağırlıkları güncellemek için basit bir Pytree gezintisi yeterli.
# ---------------------------------------------------------------------------
@jax.jit
def train_step(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    new_params = jax.tree_util.tree_map(
        lambda p, g: p - LEARNING_RATE * g, params, grads
    )
    return new_params, loss


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
# Benchmark yardımcıları (Deney 4 ile aynı; deneyler arası taşınabilir)
# ---------------------------------------------------------------------------
def choose_inner_loops(
    single_call_seconds: float,
    target_seconds: float = TARGET_SECONDS,
    min_number: int = MIN_INNER_LOOPS,
    max_number: int = MAX_INNER_LOOPS,
) -> int:
    """Her timeit tekrarının ~`target_seconds` sürmesi için gereken iç döngü sayısı.

    Bu sayı YAPILAN İŞİ değiştirmez; aynı eğitim adımı `number` kez tekrarlanır
    ve toplam süre `number`'a bölünerek adım başına süreye normalize edilir.
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

    fn'in çağrılmadan ÖNCE sıcak olması beklenir (JAX için ayrıca ölçülen first
    call, transfer yolları için untimed warmup). Dönen süreler ADIM BAŞINA'dır.
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
        f"    {label:<26}: {mean_ms:.{decimals}f} ± {std_ms:.{decimals}f} ms "
        f"(repeats={len(stats['times'])}, inner loops={stats['number']})"
    )


def relative_performance(baseline: float, candidate: float) -> str:
    if candidate <= baseline:
        return f"{baseline / candidate:.1f}x daha hızlı"
    return f"{candidate / baseline:.1f}x daha yavaş"


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 5: Pytree ile Saf Fonksiyonel Eğitim Adımı")
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    print("\nBenchmark ayarları:")
    print(f"  repeats               = {REPEATS}")
    print(f"  hedef ölçüm penceresi ~= {TARGET_SECONDS * 1000:.0f} ms / repeat")
    print("  raporlanan süre       = tek bir eğitim adımı başına")

    key = jax.random.PRNGKey(0)
    key, params_key, data_key = jax.random.split(key, 3)

    params = init_params(params_key)

    # Basit sentetik veri: y = gerçek bir doğrusal ilişki + gürültü
    x_key, noise_key = jax.random.split(data_key)
    X = jax.random.normal(x_key, (N_SAMPLES, IN_DIM))
    true_w = jnp.array([[1.0], [-2.0], [0.5], [0.3]])
    Y = X @ true_w + 0.1 * jax.random.normal(noise_key, (N_SAMPLES, OUT_DIM))

    print("\nPytree yapısı (ağırlık şekilleri):")
    for name, value in params.items():
        print(f"  {name:<4} : shape={value.shape}")

    initial_loss = float(loss_fn(params, X, Y))
    print(f"\nBaşlangıç kaybı (loss): {initial_loss:.6f}")

    # End-to-end ölçümü için host (NumPy) kopyaları: `jax.device_put` bir
    # Pytree'yi (params gibi iç içe dict) TEK ÇAĞRIDA hedef cihaza kopyalar.
    params_host = jax.tree_util.tree_map(np.asarray, params)
    X_host = np.asarray(X)
    Y_host = np.asarray(Y)

    cpu = None
    gpu_compute = None
    gpu_e2e = None
    inner_loops = {}

    for name, device in devices.items():
        print(f"\n  --- {name} ---")

        if name != "GPU":
            # Tek cihaz ölçümü (CPU): veri bir kez cihaza taşınır, sonrasında
            # SADECE derlenmiş fonksiyonun çalışma süresi ölçülür.
            params_dev = jax.device_put(params, device)
            X_dev = jax.device_put(X, device)
            Y_dev = jax.device_put(Y, device)

            # İlk çağrı: tracing + JIT derlemesi + ilk çalıştırma. number=1 ile
            # ölçülür; iç döngü UYGULANMAZ ve steady-state kalibrasyonuna
            # karıştırılmaz. Aynı zamanda warmup görevi görür.
            cpu_first_call = timeit.timeit(
                lambda: jax.block_until_ready(train_step(params_dev, X_dev, Y_dev)), number=1
            )
            print(f"    {'first call (compile+run)':<26}: {fmt_ms(cpu_first_call)}")

            # Her iç döngü çağrısı AYNI başlangıç parametreleriyle çalışır;
            # dönen new_params bir sonraki çağrıya beslenmez (ardışık eğitim
            # değil, TEK adımın maliyeti ölçülüyor).
            cpu = benchmark_steady(
                lambda: jax.block_until_ready(train_step(params_dev, X_dev, Y_dev))
            )
            inner_loops["JAX CPU"] = cpu["number"]
            print_stats("steady-state", cpu)

            new_params, _ = cpu["result"]
            loss_after_update = float(loss_fn(new_params, X_dev, Y_dev))
            print(f"    {'1 adım sonrası kayıp':<26}: {loss_after_update:.6f}")
            continue

        # =====================================================================
        # GPU: iki ayrı, birbirine KARIŞTIRILMAMASI gereken ölçüm.
        # =====================================================================

        # --- Compute-only: veri ÖNCEDEN GPU'ya taşınır ve iç döngü boyunca
        # ORADA KALIR. Host<->GPU transferi ölçüme dahil DEĞİL. ---
        params_gpu = jax.device_put(params, device)
        X_gpu = jax.device_put(X, device)
        Y_gpu = jax.device_put(Y, device)

        # CPU ile metodolojik tutarlılık: GPU'nun ilk çağrısı da ayrıca ölçülür
        # (GPU backend'i için ayrı bir derleme yapılır).
        gpu_first_call = timeit.timeit(
            lambda: jax.block_until_ready(train_step(params_gpu, X_gpu, Y_gpu)), number=1
        )
        print(f"    {'first call (compile+run)':<26}: {fmt_ms(gpu_first_call)}")

        gpu_compute = benchmark_steady(
            lambda: jax.block_until_ready(train_step(params_gpu, X_gpu, Y_gpu))
        )
        inner_loops["GPU compute-only"] = gpu_compute["number"]
        print_stats("compute-only", gpu_compute)

        new_params, _ = gpu_compute["result"]
        loss_after_update = float(loss_fn(new_params, X_gpu, Y_gpu))
        print(f"    {'1 adım sonrası kayıp':<26}: {loss_after_update:.6f}")

        # --- End-to-end (round-trip): host -> GPU -> hesapla -> host.
        # Her iç döngü çağrısı GERÇEK bir round-trip içerir (device_put ölçümün
        # içinde kalır; dışarı alınırsa bu ölçüm compute-only'ye dönüşürdü).
        # jax.device_get() hem hesabın bitmesini bekler hem sonucu host belleğine
        # kopyalar; bu yüzden ayrıca block_until_ready() eklenmez.
        # Fonksiyon zaten derlenmiş; JIT bu ölçüme girmez. ---
        def round_trip():
            params_gpu_iter = jax.device_put(params_host, device)
            X_gpu_iter = jax.device_put(X_host, device)
            Y_gpu_iter = jax.device_put(Y_host, device)
            new_params_iter, loss_iter = train_step(params_gpu_iter, X_gpu_iter, Y_gpu_iter)
            return jax.device_get((new_params_iter, loss_iter))

        round_trip()  # untimed warmup: allocator/transfer yollarını ısıt (JIT zaten hazır)
        gpu_e2e = benchmark_steady(round_trip)
        inner_loops["GPU end-to-end"] = gpu_e2e["number"]
        print_stats("end-to-end", gpu_e2e)

    # -----------------------------------------------------------------------
    # Özet
    # -----------------------------------------------------------------------
    print("\nKullanılan iç döngü (inner loop) sayıları — kalibrasyonla seçildi:")
    for label, number in inner_loops.items():
        print(f"  {label:<24}: {number}")

    if cpu is not None and (gpu_compute is not None or gpu_e2e is not None):
        print("\nGöreli performans (JAX CPU steady-state referans alınarak):")
        print("  NOT: bu deneyin amacı GPU hızlandırması değil; model bu ölçekte")
        print("       GPU'nun avantaj göstermesi için çok küçüktür.")
        if gpu_compute is not None:
            print(f"  {'GPU compute-only':<24}: {relative_performance(cpu['mean'], gpu_compute['mean'])}")
        if gpu_e2e is not None:
            print(f"  {'GPU end-to-end':<24}: {relative_performance(cpu['mean'], gpu_e2e['mean'])}")

    print("=" * 62)
