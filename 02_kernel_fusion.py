"""
Deney 2: Bellek ve Kernel Fusion Kıyaslaması
==============================================
İşlem: y = sin(A) * cos(B) + exp(A)   (10 milyon elemanlı diziler üzerinde)

NumPy her işlemi (sin, cos, exp, çarpma, toplama) ayrı bir adımda çalıştırır;
her adım sonucu belleğe yazıp tekrar okur (ayrı "kernel" çağrıları).

JAX + @jax.jit ise XLA derleyicisi sayesinde bu işlemleri tek bir
"füzyonlanmış" (fused) kernel'e dönüştürür: ara sonuçlar belleğe yazılmadan
doğrudan bir sonraki işleme aktarılır. Bu da özellikle büyük dizilerde
belirgin bir hız avantajı sağlar.
"""

import time

import numpy as np

import jax
import jax.numpy as jnp


N = 10_000_000


# ---------------------------------------------------------------------------
# NumPy versiyonu
# ---------------------------------------------------------------------------
def numpy_chain(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sin(A) * np.cos(B) + np.exp(A)


# ---------------------------------------------------------------------------
# JAX versiyonu (JIT ile derlenmiş / füzyonlanmış)
# ---------------------------------------------------------------------------
@jax.jit
def jax_chain(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(A) * jnp.cos(B) + jnp.exp(A)


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
# Belirli bir cihazda jax_chain'i çalıştırır (warmup + gerçek ölçüm).
# `jax.default_device` bloğu içinde oluşturulan diziler o cihaza yerleşir.
# ---------------------------------------------------------------------------
def run_on_device(device, A_np: np.ndarray, B_np: np.ndarray):
    with jax.default_device(device):
        A_jax = jnp.asarray(A_np)
        B_jax = jnp.asarray(B_np)

        _ = jax_chain(A_jax, B_jax).block_until_ready()  # warmup

        t0 = time.perf_counter()
        y_jax = jax_chain(A_jax, B_jax).block_until_ready()
        elapsed = time.perf_counter() - t0

    return elapsed, y_jax


if __name__ == "__main__":
    print("=" * 62)
    print(f" DENEY 2: Kernel Fusion Kıyaslaması ({N:,} eleman)".replace(",", "."))
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    rng = np.random.default_rng(42)
    A_np = rng.random(N, dtype=np.float64)
    B_np = rng.random(N, dtype=np.float64)

    # --- Referans: NumPy (cihazdan bağımsız, bir kez ölçülür) ---
    t0 = time.perf_counter()
    y_np = numpy_chain(A_np, B_np)
    numpy_time = time.perf_counter() - t0

    print(f"\n  {'NumPy (füzyonsuz)':<28} {numpy_time*1000:>10.2f} ms")

    # --- Her bulunan cihazda (CPU, varsa GPU) JAX @jax.jit çalıştır ---
    for name, device in devices.items():
        jax_time, y_jax = run_on_device(device, A_np, B_np)
        max_abs_diff = float(np.max(np.abs(y_np - np.asarray(y_jax))))

        print(f"\n  --- {name} ---")
        print(f"  {'JAX @jax.jit (füzyonlu) [' + name + ']':<28} {jax_time*1000:>10.2f} ms")
        print(f"  Hızlanma (NumPy'a göre)     : {numpy_time / jax_time:.1f}x")
        print(f"  Maks. fark (sayısal kontrol): {max_abs_diff:.2e}")

    print("=" * 62)
