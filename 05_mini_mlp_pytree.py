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
"""

import time

import numpy as np

import jax
import jax.numpy as jnp


IN_DIM = 4
HIDDEN_DIM = 8
OUT_DIM = 1
N_SAMPLES = 256
LEARNING_RATE = 0.1


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


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 5: Pytree ile Saf Fonksiyonel Eğitim Adımı")
    print("=" * 62)

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

    # --- warmup: ilk çağrı derlemeyi (tracing/compilation) içerir ---
    warm_params, warm_loss = train_step(params, X, Y)
    jax.block_until_ready((warm_params, warm_loss))

    initial_loss = float(loss_fn(params, X, Y))

    # --- gerçek ölçüm: sadece derlenmiş fonksiyonun çalışma süresi ---
    t0 = time.perf_counter()
    new_params, loss_after_step = train_step(params, X, Y)
    jax.block_until_ready((new_params, loss_after_step))
    t1 = time.perf_counter()
    step_time = t1 - t0

    # loss_after_step, GÜNCELLEMEDEN ÖNCEKİ (eski) ağırlıklarla hesaplanan kayıptır
    # (jax.value_and_grad böyle çalışır); güncelleme sonrası gerçek kaybı görmek için
    # yeni ağırlıklarla (new_params) tekrar hesaplıyoruz.
    loss_after_update = float(loss_fn(new_params, X, Y))

    print(f"\nBaşlangıç kaybı (loss)        : {initial_loss:.6f}")
    print(f"1 adım sonrası kayıp (loss)   : {loss_after_update:.6f}")
    print(f"Tek eğitim adımı süresi       : {step_time*1000:.3f} ms  (jit sonrası)")
    print("=" * 62)
