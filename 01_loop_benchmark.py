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
# Belirli bir cihazda fori_loop ve scan'i çalıştırıp (warmup + ölçüm) süreleri döndürür.
# `jax.default_device` bloğu içinde oluşturulan/çalıştırılan her şey o cihazda çalışır.
# ---------------------------------------------------------------------------
def run_on_device(device) -> dict:
    with jax.default_device(device):
        _ = jax_fori_loop(N_STEPS, DT).block_until_ready()  # warmup
        t0 = time.perf_counter()
        fori_result = jax_fori_loop(N_STEPS, DT).block_until_ready()
        fori_time = time.perf_counter() - t0

        _ = jax_scan_loop(N_STEPS, DT).block_until_ready()  # warmup
        t0 = time.perf_counter()
        scan_result = jax_scan_loop(N_STEPS, DT).block_until_ready()
        scan_time = time.perf_counter() - t0

    return {
        "fori_time": fori_time, "fori_result": float(fori_result),
        "scan_time": scan_time, "scan_result": float(scan_result),
    }


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 1: Döngü Karşılaştırması (1.000.000 adım)")
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    # --- Referans: Saf Python (cihazdan bağımsız, bir kez ölçülür) ---
    t0 = time.perf_counter()
    py_result = python_for_loop(N_STEPS, DT)
    py_time = time.perf_counter() - t0

    print("\nSonuçlar (warmup sonrası, saf çalışma süresi):\n")
    print_row("Python for döngüsü", py_time, py_result)

    # --- Her bulunan cihazda (CPU, varsa GPU) fori_loop ve scan'i çalıştır ---
    device_results = {}
    for name, device in devices.items():
        print(f"\n  --- {name} ---")
        r = run_on_device(device)
        device_results[name] = r
        print_row(f"jax.lax.fori_loop [{name}]", r["fori_time"], r["fori_result"])
        print_row(f"jax.lax.scan [{name}]", r["scan_time"], r["scan_result"])

    print("\nHızlanma (Python'a göre):")
    for name, r in device_results.items():
        print(f"  [{name}] fori_loop : {py_time / r['fori_time']:>8.1f}x daha hızlı")
        print(f"  [{name}] scan      : {py_time / r['scan_time']:>8.1f}x daha hızlı")
    print("=" * 62)
