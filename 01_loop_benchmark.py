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

Her yöntem `timeit` ile 10 kez ölçülür; ortalama ve standart sapma raporlanır.
JAX fonksiyonları için JIT derleme süresi bir warmup çağrısıyla ölçüm dışı bırakılır
ve her ölçüm çağrısı `.block_until_ready()` ile senkronize edilir.

Not (ilk çağrı / steady-state ayrımı — Deney 2'deki ile aynı mantık): Python
için ölçümden önce 1 kez untimed warmup çalıştırılır (Python'da ölçülecek bir
"derleme" maliyeti yoktur). JAX'ta ise `fori_loop`/`scan`'in ilk çağrısı hem
JIT derlemesini hem de gerçek çalışmayı içerir; bu çağrı warmup olarak
kullanılmaya devam eder, ama artık atılmıyor — süresi "first call (compile +
run)" olarak ayrıca raporlanır. Sonrasındaki 10 tekrarlık "steady-state"
ölçümü, derlenmiş fonksiyonun saf çalışma süresini yansıtır. Hızlanma
(speedup) hesabı SADECE steady-state ortalamaları üzerinden yapılır. Ayrıca
her JAX yöntemi için, ilk çağrının derleme maliyetinin kaç tekrarda "amorti"
olduğunu (break-even) gösteren yaklaşık bir tekrar sayısı raporlanır.

Not (compute-only / end-to-end ayrımı burada YOK): `N_STEPS` ve `DT` sabit,
küçük skaler değerlerdir (N_STEPS zaten `static_argnums` ile derleme zamanı
sabitidir); ölçülecek büyüklükte bir host dizisi cihaza taşınmaz. Bu yüzden
buradaki tüm ölçümler doğası gereği "compute-only"dur; Deney 2, 3, 5 ve 6'da
olduğu gibi ayrı bir "end-to-end" varyantı anlamlı değildir.

Kural: state_{n+1} = state_n + sin(n * 0.001)
"""

import math
import statistics
import timeit
from functools import partial

import jax
import jax.numpy as jnp

N_STEPS = 1_000_000
DT = 0.001
REPEATS = 10


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


def print_stats(label: str, times: list, result: float) -> None:
    mean_ms = statistics.mean(times) * 1000
    std_ms = statistics.stdev(times) * 1000
    print(f"  {label}: {mean_ms:.2f} ± {std_ms:.2f} ms (n={len(times)})   sonuç = {result:.6f}")


# ---------------------------------------------------------------------------
# Kaç tekrardan sonra JAX'in TOPLAM süresi (ilk çağrı + sonraki steady-state
# çağrılar) Python baseline'ının toplam süresini geçer (amortisman noktası)?
#   first_call + (n-1)*steady <= n*baseline  =>  n >= (first_call-steady)/(baseline-steady)
# ---------------------------------------------------------------------------
def break_even_repeats(first_call_time: float, steady_mean: float, baseline_mean: float):
    if baseline_mean <= steady_mean:
        return None  # steady-state Python'dan hızlı değil; hiçbir zaman amorti olmaz
    n = (first_call_time - steady_mean) / (baseline_mean - steady_mean)
    return max(1, math.ceil(n))


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 1: Döngü Karşılaştırması (1.000.000 adım)")
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    # --- Referans: Saf Python (cihazdan bağımsız) ---
    python_for_loop(N_STEPS, DT)  # untimed warmup (Python'da derleme maliyeti yok)

    py_times, py_result = time_it(lambda: python_for_loop(N_STEPS, DT))
    py_mean = statistics.mean(py_times)

    print(f"\nSonuçlar (ortalama ± std, n={REPEATS}):\n")
    print_stats("Python for steady-state", py_times, py_result)

    # --- Her bulunan cihazda (CPU, varsa GPU) fori_loop ve scan'i ölç ---
    device_means = {}
    for name, device in devices.items():
        print(f"\n  --- {name} ---")
        with jax.default_device(device):
            # İlk çağrı: JIT derlemesi + ilk çalıştırmayı birlikte içerir. Bu
            # çağrı warmup görevini de görür (sonraki çağrılar zaten derlenmiş
            # olacak), ama artık atılmıyor — süresi ayrıca raporlanıyor.
            fori_first_call = timeit.timeit(lambda: jax_fori_loop(N_STEPS, DT).block_until_ready(), number=1)
            scan_first_call = timeit.timeit(lambda: jax_scan_loop(N_STEPS, DT).block_until_ready(), number=1)

            print(f"  JAX fori_loop first call (compile+run) [{name}] : {fori_first_call*1000:.2f} ms")
            print(f"  JAX scan first call (compile+run) [{name}]      : {scan_first_call*1000:.2f} ms")

            fori_times, fori_result = time_it(lambda: jax_fori_loop(N_STEPS, DT).block_until_ready())
            scan_times, scan_result = time_it(lambda: jax_scan_loop(N_STEPS, DT).block_until_ready())

        print_stats(f"jax.lax.fori_loop steady-state [{name}]", fori_times, float(fori_result))
        print_stats(f"jax.lax.scan steady-state [{name}]", scan_times, float(scan_result))

        device_means[name] = {
            "fori": statistics.mean(fori_times),
            "scan": statistics.mean(scan_times),
            "fori_first_call": fori_first_call,
            "scan_first_call": scan_first_call,
        }

    print("\nHızlanma (Python'a göre, SADECE steady-state ortalamaları üzerinden):")
    for name, means in device_means.items():
        print(f"  [{name}] fori_loop : {py_mean / means['fori']:>8.1f}x daha hızlı")
        print(f"  [{name}] scan      : {py_mean / means['scan']:>8.1f}x daha hızlı")

    print("\nBreak-even (ilk çağrının derleme maliyeti kaç tekrarda amorti oluyor):")
    for name, means in device_means.items():
        fori_be = break_even_repeats(means["fori_first_call"], means["fori"], py_mean)
        scan_be = break_even_repeats(means["scan_first_call"], means["scan"], py_mean)
        fori_be_str = f"~{fori_be} tekrar" if fori_be is not None else "asla (steady-state Python'dan yavaş)"
        scan_be_str = f"~{scan_be} tekrar" if scan_be is not None else "asla (steady-state Python'dan yavaş)"
        print(f"  [{name}] fori_loop break-even: {fori_be_str}")
        print(f"  [{name}] scan break-even     : {scan_be_str}")

    print("=" * 62)
