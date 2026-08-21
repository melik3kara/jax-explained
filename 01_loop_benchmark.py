"""
Deney 1: Döngü Karşılaştırması
================================
1 milyon adımlık kümülatif bir durum güncellemesi (ardışık sinüs toplamı)
üç farklı şekilde hesaplanır:

  1) Saf Python `for` döngüsü      -> yorumlanır, adım adım çok yavaş.
  2) `jax.lax.fori_loop`           -> JIT ile tek bir derlenmiş döngüye dönüşür.
  3) `jax.lax.scan`                -> aynı işi "tarama" (scan) ile yapar,
                                       genelde fori_loop'tan bile hızlıdır
                                       çünkü XLA daha fazla optimizasyon fırsatı bulur.

Kural: state_{n+1} = state_n + sin(n * 0.001)
"""

import math
import time
from functools import partial

import jax
import jax.numpy as jnp


N_STEPS = 1_000_000
DT = 0.001


# ---------------------------------------------------------------------------
# 1) Saf Python for döngüsü (referans / baseline)
# ---------------------------------------------------------------------------
def python_for_loop(n_steps: int, dt: float) -> float:
    state = 0.0
    for i in range(n_steps):
        state = state + math.sin(i * dt)
    return state


# ---------------------------------------------------------------------------
# 2) jax.lax.fori_loop
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(0,))
def jax_fori_loop(n_steps: int, dt: float) -> jnp.ndarray:
    def body_fun(i, state):
        return state + jnp.sin(i * dt)

    return jax.lax.fori_loop(0, n_steps, body_fun, 0.0)


# ---------------------------------------------------------------------------
# 3) jax.lax.scan
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(0,))
def jax_scan_loop(n_steps: int, dt: float) -> jnp.ndarray:
    xs = jnp.arange(n_steps)

    def step(carry, i):
        new_carry = carry + jnp.sin(i * dt)
        return new_carry, None  # ikinci eleman: her adımda biriktirilecek çıktı (burada yok)

    final_state, _ = jax.lax.scan(step, 0.0, xs)
    return final_state


# ---------------------------------------------------------------------------
# Yardımcı: şık çıktı basma
# ---------------------------------------------------------------------------
def print_row(label: str, seconds: float, result: float) -> None:
    print(f"  {label:<32} {seconds*1000:>10.2f} ms   sonuç = {result:.6f}")


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 1: Döngü Karşılaştırması (1.000.000 adım)")
    print("=" * 62)

    # --- 1) Saf Python ---
    t0 = time.perf_counter()
    py_result = python_for_loop(N_STEPS, DT)
    t1 = time.perf_counter()
    py_time = t1 - t0

    # --- 2) fori_loop: warmup (derleme) + gerçek ölçüm ---
    _ = jax_fori_loop(N_STEPS, DT).block_until_ready()  # warmup, süresi sayılmaz

    t0 = time.perf_counter()
    fori_result = jax_fori_loop(N_STEPS, DT).block_until_ready()
    t1 = time.perf_counter()
    fori_time = t1 - t0

    # --- 3) scan: warmup (derleme) + gerçek ölçüm ---
    _ = jax_scan_loop(N_STEPS, DT).block_until_ready()  # warmup, süresi sayılmaz

    t0 = time.perf_counter()
    scan_result = jax_scan_loop(N_STEPS, DT).block_until_ready()
    t1 = time.perf_counter()
    scan_time = t1 - t0

    print("\nSonuçlar (warmup sonrası, saf çalışma süresi):\n")
    print_row("Python for döngüsü", py_time, py_result)
    print_row("jax.lax.fori_loop", fori_time, float(fori_result))
    print_row("jax.lax.scan", scan_time, float(scan_result))

    print("\nHızlanma (Python'a göre):")
    print(f"  fori_loop : {py_time / fori_time:>8.1f}x daha hızlı")
    print(f"  scan      : {py_time / scan_time:>8.1f}x daha hızlı")
    print("=" * 62)
