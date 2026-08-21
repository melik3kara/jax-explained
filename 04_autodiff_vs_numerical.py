"""
Deney 4: Türev Hassasiyeti ve Hızı (Autodiff vs Sonlu Farklar)
=================================================================
Çok değişkenli fonksiyon:  f(x0, x1, x2) = x0^2 * x1 + sin(x2) * x0 + x1 * x2^2

Bu fonksiyonun gradyanı (1. türev) ve Hessian matrisi (2. türev) iki yöntemle
hesaplanır:

  1) NumPy ile Sonlu Farklar (Finite Differences)
     -> Türevin TANIMINA yakın bir yaklaşıklama:  f'(x) ~ (f(x+h) - f(x-h)) / 2h
     -> Adım büyüklüğü `h`'a bağlı yuvarlama/kesme hataları içerir.
     -> Hessian için "türevin türevini" almak gerektiğinden O(n^2) fonksiyon
        çağrısı gerekir; yavaştır.

  2) JAX ile Otomatik Türev (Automatic Differentiation)
     -> `jax.grad`      : gradyanı MAKİNE HASSASİYETİNDE, analitik olarak hesaplar.
     -> `jacfwd(jacrev(f))`: ileri-geri mod kombinasyonuyla Hessian'ı verimli
        şekilde hesaplar (Hessian hesaplamak için önerilen standart JAX deseni).

Otomatik türev, sayısal (finite difference) türevden FARKLI bir şeydir:
Yaklaşıklama yapmaz, zincir kuralını (chain rule) kodun kendisi üzerinde
otomatik olarak uygular ve tam (exact) sonuç üretir.
"""

import time

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # hassas kıyaslama için float64


X0 = jnp.array([1.5, -2.0, 0.7])  # (x0, x1, x2) değerlendirme noktası


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


# ---------------------------------------------------------------------------
# 2) JAX ile Otomatik Türev
# ---------------------------------------------------------------------------
jax_grad_fn = jax.jit(jax.grad(f))
jax_hessian_fn = jax.jit(jax.jacfwd(jax.jacrev(f)))  # Hessian için standart JAX deseni


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
# Belirli bir cihazda gradyan+Hessian hesabını çalıştırır (warmup + ölçüm).
# ÖNEMLİ: X0 modül seviyesinde bir kez oluşturulduğu için belirli bir cihaza
# "commit" edilmiştir; sadece `jax.default_device` bağlamına girmek onu
# BAŞKA bir cihaza TAŞIMAZ. Bu yüzden `jax.device_put(X0, device)` ile
# diziyi HER cihaz için AÇIKÇA o cihaza kopyalıyoruz.
# ---------------------------------------------------------------------------
def run_on_device(device):
    x0_dev = jax.device_put(X0, device)

    _ = jax_grad_fn(x0_dev).block_until_ready()  # warmup
    _ = jax_hessian_fn(x0_dev).block_until_ready()  # warmup

    t0 = time.perf_counter()
    grad_jax = jax_grad_fn(x0_dev).block_until_ready()
    hess_jax = jax_hessian_fn(x0_dev).block_until_ready()
    elapsed = time.perf_counter() - t0

    return elapsed, grad_jax, hess_jax


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 4: Autodiff vs Sonlu Farklar (Gradyan & Hessian)")
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    x_np = np.array(X0, dtype=np.float64)

    # --- Referans: NumPy sonlu farklar (cihazdan bağımsız, bir kez ölçülür) ---
    t0 = time.perf_counter()
    grad_numerical = numerical_gradient(x_np)
    hess_numerical = numerical_hessian(x_np)
    numerical_time = time.perf_counter() - t0

    print("\nGradyan (df/dx0, df/dx1, df/dx2):")
    print(f"  Sonlu Farklar : {grad_numerical}")

    print("\nHessian matrisi (Sonlu Farklar):")
    print("   ", str(hess_numerical).replace("\n", "\n    "))

    print(f"\n  NumPy Sonlu Farklar süresi: {numerical_time*1000:.2f} ms")

    # --- Her bulunan cihazda (CPU, varsa GPU) JAX autodiff çalıştır ---
    for name, device in devices.items():
        jax_time, grad_jax, hess_jax = run_on_device(device)

        grad_diff = float(np.max(np.abs(grad_numerical - np.asarray(grad_jax))))
        hess_diff = float(np.max(np.abs(hess_numerical - np.asarray(hess_jax))))

        print(f"\n  --- {name} ---")
        print(f"  jax.grad      : {np.asarray(grad_jax)}  (fark: {grad_diff:.2e})")
        print("  jacfwd(jacrev(f)) :")
        print("   ", str(np.asarray(hess_jax)).replace("\n", "\n    "))
        print(f"  Hessian maks. fark : {hess_diff:.2e}")
        print(f"  JAX Autodiff süresi [{name}] : {jax_time*1000:.2f} ms")
        print(f"  Hızlanma (NumPy'a göre)      : {numerical_time / jax_time:.1f}x")

    print("=" * 62)
