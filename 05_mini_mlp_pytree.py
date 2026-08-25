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

GPU'da iki ayrı süre raporlanır (compute-only vs end-to-end); nedeni ve
kapsamı için Deney 2'deki (02_kernel_fusion.py) açıklamaya bakınız. Burada
girdi/ağırlıklar küçük olduğundan transfer maliyeti ihmal edilebilir
düzeydedir; yine de metodoloji tutarlılığı için aynı ayrım uygulanır.

Tek eğitim adımı `timeit` ile 10 kez ölçülür; ortalama ve standart sapma
raporlanır. JIT derleme süresi bir warmup çağrısıyla ölçüm dışı bırakılır ve
her ölçüm çağrısı `.block_until_ready()` ile senkronize edilir.
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
    print(" DENEY 5: Pytree ile Saf Fonksiyonel Eğitim Adımı")
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

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

    for name, device in devices.items():
        print(f"\n  --- {name} ---")

        if name != "GPU":
            # Tek cihaz ölçümü (CPU): veri bir kez cihaza taşınır, sonrasında
            # SADECE derlenmiş fonksiyonun çalışma süresi ölçülür.
            params_dev = jax.device_put(params, device)
            X_dev = jax.device_put(X, device)
            Y_dev = jax.device_put(Y, device)

            # İlk çağrı: JIT derlemesi + ilk çalıştırmayı birlikte içerir. Bu
            # çağrı warmup görevini de görür, ama artık atılmıyor — süresi
            # ayrıca raporlanıyor (bkz. Deney 2'deki aynı ayrım).
            first_call_time = timeit.timeit(
                lambda: jax.block_until_ready(train_step(params_dev, X_dev, Y_dev)), number=1
            )
            print(f"  {'JAX first call (compile+run)':<28}: {first_call_time*1000:6.3f} ms")

            cpu_times, (new_params, _) = time_it(
                lambda: jax.block_until_ready(train_step(params_dev, X_dev, Y_dev))
            )
            loss_after_update = float(loss_fn(new_params, X_dev, Y_dev))

            print(f"  1 adım sonrası kayıp (loss)    : {loss_after_update:.6f}")
            print_stats(f"Tek eğitim adımı steady-state [{name}]", cpu_times)
            continue

        # =====================================================================
        # GPU: iki ayrı, birbirine KARIŞTIRILMAMASI gereken ölçüm.
        # =====================================================================

        # --- Compute-only: veri ÖNCEDEN GPU'ya taşınmış, warmup yapılmış.
        # Host<->GPU transferi ölçüme dahil DEĞİL. ---
        params_gpu = jax.device_put(params, device)
        X_gpu = jax.device_put(X, device)
        Y_gpu = jax.device_put(Y, device)

        jax.block_until_ready(train_step(params_gpu, X_gpu, Y_gpu))  # warmup (JIT derlemesi)

        gpu_compute_times, (new_params, _) = time_it(
            lambda: jax.block_until_ready(train_step(params_gpu, X_gpu, Y_gpu))
        )
        loss_after_update = float(loss_fn(new_params, X_gpu, Y_gpu))

        print(f"  1 adım sonrası kayıp (loss)    : {loss_after_update:.6f}")
        print_stats("GPU compute-only", gpu_compute_times)

        # --- End-to-end (round-trip): host -> GPU -> hesapla -> host.
        # jax.device_get() hem hesaplamanın bitmesini bekler hem sonucu gerçekten
        # host belleğine kopyalar. Fonksiyon zaten derlenmiş; JIT bu ölçüme girmez. ---
        def round_trip():
            params_gpu_iter = jax.device_put(params_host, device)
            X_gpu_iter = jax.device_put(X_host, device)
            Y_gpu_iter = jax.device_put(Y_host, device)
            new_params_iter, loss_iter = train_step(params_gpu_iter, X_gpu_iter, Y_gpu_iter)
            return jax.device_get((new_params_iter, loss_iter))

        round_trip()  # warmup: allocator/transfer yollarını ısıt (JIT zaten hazır)
        gpu_e2e_times, _ = time_it(round_trip)
        print_stats("GPU end-to-end", gpu_e2e_times)

    print("=" * 62)
