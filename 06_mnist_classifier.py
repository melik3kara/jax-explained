"""
Deney 6 (Bonus): Basit MNIST Sınıflandırma
=============================================
Önceki deneylerde öğrenilen kavramları (Pytree, jax.grad, @jax.jit) gerçek
bir veri seti üzerinde birleştiren uçtan uca, küçük bir örnek.

Aynı 1 gizli katmanlı MLP (784 -> 128 -> 10) iki farklı şekilde eğitilir:

  1) "Klasik" yöntem: NumPy ile ELLE türetilmiş backpropagation.
     Softmax + Cross-Entropy türevlerini kendimiz matematiksel olarak
     çıkarıp kodladık (dW2, db2, dW1, db1). Hataya açık, fonksiyon
     değişirse türevleri baştan türetmek gerekir.

  2) JAX yöntemi: `jax.grad` ile OTOMATİK türev + `@jax.jit` ile derleme.
     Sadece ileri yayılımı (forward pass) ve kayıp fonksiyonunu yazdık;
     geri yayılım (backward pass) otomatik olarak, tam ve hatasız
     hesaplanıyor.

İki yöntem de AYNI başlangıç ağırlıklarından, aynı veri sırasıyla eğitilir;
böylece hem doğruluk (accuracy) hem de eğitim süresi adil bir şekilde
kıyaslanabilir.

Not: İlk çalıştırmada MNIST verisi (~11 MB) internetten indirilip
`.mnist_cache/` klasörüne kaydedilir; sonraki çalıştırmalarda diskten okunur.
"""

import os
import time
import urllib.request

import numpy as np

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
N_TRAIN = 10_000
N_TEST = 2_000
HIDDEN_DIM = 128
BATCH_SIZE = 100
EPOCHS = 5
LEARNING_RATE = 0.5

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mnist_cache", "mnist.npz")


# ---------------------------------------------------------------------------
# Veri: indir (bir kere) + önbellekten oku + ön işle
# ---------------------------------------------------------------------------
def load_mnist():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if not os.path.exists(CACHE_PATH):
        print(f"  MNIST verisi indiriliyor (~11 MB)...\n  Kaynak: {MNIST_URL}")
        urllib.request.urlretrieve(MNIST_URL, CACHE_PATH)
        print("  İndirme tamamlandı, önbelleğe kaydedildi.")
    else:
        print(f"  MNIST verisi önbellekten okunuyor: {CACHE_PATH}")

    with np.load(CACHE_PATH) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
    return (x_train, y_train), (x_test, y_test)


def preprocess(x, y, n_samples, rng):
    idx = rng.choice(len(x), size=n_samples, replace=False)
    x = x[idx].reshape(n_samples, -1).astype(np.float32) / 255.0  # düzleştir + normalize et
    y_labels = y[idx].astype(np.int32)
    y_onehot = np.zeros((n_samples, 10), dtype=np.float32)
    y_onehot[np.arange(n_samples), y_labels] = 1.0
    return x, y_labels, y_onehot


# ---------------------------------------------------------------------------
# Ortak ağırlık ilklendirme (her iki yöntem de AYNI başlangıç noktasından başlar)
# ---------------------------------------------------------------------------
def init_params(rng):
    return {
        "W1": (rng.standard_normal((784, HIDDEN_DIM)) * np.sqrt(1.0 / 784)).astype(np.float32),
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W2": (rng.standard_normal((HIDDEN_DIM, 10)) * np.sqrt(1.0 / HIDDEN_DIM)).astype(np.float32),
        "b2": np.zeros(10, dtype=np.float32),
    }


# ===========================================================================
# 1) NumPy: elle türetilmiş backpropagation
# ===========================================================================
def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward_np(params, x):
    z1 = x @ params["W1"] + params["b1"]
    h = np.tanh(z1)
    z2 = h @ params["W2"] + params["b2"]
    return z1, h, z2


def loss_np(params, x, y_onehot):
    _, _, z2 = forward_np(params, x)
    probs = softmax_np(z2)
    n = x.shape[0]
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))


def train_step_np(params, x, y_onehot, lr):
    n = x.shape[0]
    z1, h, z2 = forward_np(params, x)
    probs = softmax_np(z2)

    # -- elle türetilmiş zincir kuralı (chain rule) --
    dz2 = (probs - y_onehot) / n
    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0)

    dh = dz2 @ params["W2"].T
    dz1 = dh * (1 - np.tanh(z1) ** 2)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)

    new_params = {
        "W1": params["W1"] - lr * dW1,
        "b1": params["b1"] - lr * db1,
        "W2": params["W2"] - lr * dW2,
        "b2": params["b2"] - lr * db2,
    }
    loss = loss_np(params, x, y_onehot)
    return new_params, loss


def accuracy_np(params, x, y_labels):
    _, _, z2 = forward_np(params, x)
    preds = z2.argmax(axis=1)
    return float(np.mean(preds == y_labels))


# ===========================================================================
# 2) JAX: jax.grad ile otomatik türev + @jax.jit ile derleme
# ===========================================================================
def forward_jax(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def loss_jax(params, x, y_onehot):
    logits = forward_jax(params, x)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(y_onehot * log_probs, axis=1))


@jax.jit
def train_step_jax(params, x, y_onehot, lr):
    loss, grads = jax.value_and_grad(loss_jax)(params, x, y_onehot)  # backward pass OTOMATİK
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


@jax.jit
def accuracy_jax(params, x, y_labels):
    logits = forward_jax(params, x)
    preds = jnp.argmax(logits, axis=1)
    return jnp.mean(preds == y_labels)


# ---------------------------------------------------------------------------
# Yardımcı: bir epoch'luk mini-batch indekslerini üret (son eksik parça atılır)
# ---------------------------------------------------------------------------
def batch_indices(n_samples, batch_size, rng):
    perm = rng.permutation(n_samples)
    n_batches = n_samples // batch_size
    for i in range(n_batches):
        yield perm[i * batch_size:(i + 1) * batch_size]


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
# Belirli bir cihazda tam JAX eğitim döngüsünü (5 epoch) çalıştırır.
# `jax.device_put` ile ağırlıkları ve veriyi AÇIKÇA o cihaza taşıyoruz;
# aksi halde önceden oluşturulmuş diziler "commit" edildikleri ilk cihazda
# kalmaya devam eder.
# ---------------------------------------------------------------------------
def run_jax_training(device, name, init, x_train, y_train_onehot, x_test, y_test_labels):
    params_jax = jax.device_put(init, device)
    x_train_jax = jax.device_put(x_train, device)
    y_train_onehot_jax = jax.device_put(y_train_onehot, device)
    x_test_jax = jax.device_put(x_test, device)
    y_test_labels_jax = jax.device_put(y_test_labels, device)

    # --- warmup: ilk çağrı derlemeyi (tracing/compilation) içerir, süresi sayılmaz ---
    first_batch = jax.device_put(jnp.arange(BATCH_SIZE), device)
    warm_params, warm_loss = train_step_jax(
        params_jax, x_train_jax[first_batch], y_train_onehot_jax[first_batch], LEARNING_RATE
    )
    jax.block_until_ready((warm_params, warm_loss))
    jax.block_until_ready(accuracy_jax(params_jax, x_test_jax, y_test_labels_jax))

    epoch_rng = np.random.default_rng(1)  # NumPy ile AYNI batch sırası

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for idx in batch_indices(N_TRAIN, BATCH_SIZE, epoch_rng):
            idx_jax = jax.device_put(idx, device)
            params_jax, batch_loss = train_step_jax(
                params_jax, x_train_jax[idx_jax], y_train_onehot_jax[idx_jax], LEARNING_RATE
            )
            epoch_loss += float(batch_loss)
            n_batches += 1
        test_acc = float(accuracy_jax(params_jax, x_test_jax, y_test_labels_jax))
        print(f"  [{name}] Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/n_batches:.4f}  test_acc={test_acc*100:.2f}%")
    jax.block_until_ready(params_jax)
    elapsed = time.perf_counter() - t0
    final_acc = float(accuracy_jax(params_jax, x_test_jax, y_test_labels_jax))

    return elapsed, final_acc


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 6 (Bonus): Basit MNIST Sınıflandırma")
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_mnist()

    rng = np.random.default_rng(0)
    x_train, y_train_labels, y_train_onehot = preprocess(x_train_full, y_train_full, N_TRAIN, rng)
    x_test, y_test_labels, _ = preprocess(x_test_full, y_test_full, N_TEST, rng)

    print(f"\n  Eğitim örneği : {N_TRAIN:,}".replace(",", "."))
    print(f"  Test örneği   : {N_TEST:,}".replace(",", "."))
    print(f"  Mimari        : 784 -> {HIDDEN_DIM} (tanh) -> 10 (softmax)")
    print(f"  Epoch / Batch : {EPOCHS} epoch, batch boyutu {BATCH_SIZE}")

    init = init_params(np.random.default_rng(42))  # her iki yöntem de aynı başlangıçtan başlar

    # =======================================================================
    # 1) NumPy ile eğitim (elle backprop)
    # =======================================================================
    print("\n" + "-" * 62)
    print("  [1] NumPy (elle türetilmiş backpropagation)")
    print("-" * 62)

    params_np = {k: v.copy() for k, v in init.items()}
    epoch_rng = np.random.default_rng(1)

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for idx in batch_indices(N_TRAIN, BATCH_SIZE, epoch_rng):
            params_np, batch_loss = train_step_np(
                params_np, x_train[idx], y_train_onehot[idx], LEARNING_RATE
            )
            epoch_loss += batch_loss
            n_batches += 1
        test_acc = accuracy_np(params_np, x_test, y_test_labels)
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/n_batches:.4f}  test_acc={test_acc*100:.2f}%")
    t1 = time.perf_counter()
    numpy_time = t1 - t0
    numpy_final_acc = accuracy_np(params_np, x_test, y_test_labels)

    # =======================================================================
    # 2) JAX ile eğitim (otomatik türev + jit) — bulunan HER cihazda
    # =======================================================================
    device_results = {}
    for name, device in devices.items():
        print("\n" + "-" * 62)
        print(f"  [2] JAX (jax.grad + @jax.jit) — {name}")
        print("-" * 62)
        elapsed, final_acc = run_jax_training(
            device, name, init, x_train, y_train_onehot, x_test, y_test_labels
        )
        device_results[name] = (elapsed, final_acc)

    # =======================================================================
    # Özet
    # =======================================================================
    print("\n" + "=" * 62)
    print(" ÖZET")
    print("=" * 62)
    print(f"  {'Yöntem':<32}{'Süre':>12}{'Test Doğruluğu':>18}")
    print(f"  {'NumPy (elle backprop)':<32}{numpy_time:>10.2f} s{numpy_final_acc*100:>16.2f}%")
    for name, (elapsed, final_acc) in device_results.items():
        label = f"JAX (grad + jit) [{name}]"
        print(f"  {label:<32}{elapsed:>10.2f} s{final_acc*100:>16.2f}%")

    print("\n  Hızlanma (NumPy'a göre):")
    slow_devices = []
    for name, (elapsed, _) in device_results.items():
        print(f"    [{name}] : {numpy_time / elapsed:.1f}x")
        if elapsed > numpy_time:
            slow_devices.append(name)

    if slow_devices:
        print(
            f"\n  Not: JAX [{', '.join(slow_devices)}] burada NumPy'dan YAVAŞ çıkabilir —\n"
            "  bu bir hata değil, önemli bir ders! Batch boyutu (100) çok küçük\n"
            "  olduğundan, her adımdaki asıl hesaplama ucuz kalıyor; buna karşılık\n"
            "  Python döngüsünde her batch için ayrı bir jit çağrısı dispatch etme ve\n"
            "  host<->device veri transferi maliyeti, kazanılan hesaplama zamanını\n"
            "  aşıyor. JIT/JAX; büyük batch'lerde, GPU/TPU'da veya tüm epoch döngüsü\n"
            "  tek bir `jax.lax.scan` ile derlenip Python dispatch'i tamamen ortadan\n"
            "  kaldırıldığında asıl avantajını gösterir."
        )
    print("=" * 62)
