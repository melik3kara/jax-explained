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


if __name__ == "__main__":
    print("=" * 62)
    print(f" DENEY 2: Kernel Fusion Kıyaslaması ({N:,} eleman)".replace(",", "."))
    print("=" * 62)

    rng = np.random.default_rng(42)
    A_np = rng.random(N, dtype=np.float64)
    B_np = rng.random(N, dtype=np.float64)

    # --- NumPy ölçümü ---
    t0 = time.perf_counter()
    y_np = numpy_chain(A_np, B_np)
    t1 = time.perf_counter()
    numpy_time = t1 - t0

    # --- JAX: veriyi cihaza taşı, warmup (derleme) yap, sonra ölç ---
    A_jax = jnp.asarray(A_np)
    B_jax = jnp.asarray(B_np)

    _ = jax_chain(A_jax, B_jax).block_until_ready()  # warmup, süresi sayılmaz

    t0 = time.perf_counter()
    y_jax = jax_chain(A_jax, B_jax).block_until_ready()
    t1 = time.perf_counter()
    jax_time = t1 - t0

    max_abs_diff = float(np.max(np.abs(y_np - np.asarray(y_jax))))

    print(f"\n  {'NumPy (füzyonsuz)':<28} {numpy_time*1000:>10.2f} ms")
    print(f"  {'JAX @jax.jit (füzyonlu)':<28} {jax_time*1000:>10.2f} ms")
    print(f"\n  Hızlanma        : {numpy_time / jax_time:.1f}x")
    print(f"  Maks. fark       : {max_abs_diff:.2e}  (sayısal doğruluk kontrolü)")
    print("=" * 62)
