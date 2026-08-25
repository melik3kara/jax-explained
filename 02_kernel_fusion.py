"""
Deney 2: Bellek ve Kernel Fusion Kıyaslaması
==============================================
İşlem: y = sin(A) * cos(B) + exp(A)   (10 milyon elemanlı diziler üzerinde)

NumPy her matematiksel işlemi (sin, cos, exp, çarpma, toplama) için ayrı bir
CPU işlemi (ufunc) çalıştırır; her adımın sonucu belleğe yazılıp bir sonraki
adım için tekrar okunur.

JAX + @jax.jit ise XLA derleyicisi sayesinde bu işlemleri tek bir
"füzyonlanmış" (fused) kernel'e dönüştürür: ara sonuçlar belleğe yazılmadan
doğrudan bir sonraki işleme aktarılır. Bu füzyon hem CPU hem GPU'da geçerlidir;
GPU'da ayrıca veri PCIe üzerinden taşınmak zorunda olduğu için "hesaplama
süresi" ile "uçtan uca (transfer dahil) süre" arasındaki fark daha belirgin
hâle gelir — bu yüzden GPU için iki ayrı ölçüm raporluyoruz:

  * GPU compute-only : Veri ÖNCEDEN GPU'ya taşınmış durumda; sadece
                        hesaplamanın (ve GPU->host senkronizasyonunun) süresi.
  * GPU end-to-end    : Her tekrarda host (NumPy) dizilerinden başlanır,
                         GPU'ya taşınır, hesaplanır ve sonuç host'a geri
                         getirilir (round-trip).

Not (adil kıyaslama): NumPy dizileri BİLEREK float32 olarak oluşturuluyor.
JAX varsayılan olarak float64'ü float32'ye sessizce indirger (x64 modu kapalı);
NumPy tarafı float64 bırakılsaydı iki taraf farklı hassasiyette çalışır ve
hız kıyaslaması yanıltıcı olurdu.

Her yöntem `timeit` ile 10 kez ölçülür; ortalama ve standart sapma raporlanır.
JIT derleme süresi warmup çağrılarıyla ölçüm dışı bırakılır; veri oluşturma
(rastgele dizi üretimi) hiçbir ölçüme dahil değildir.

Not (ilk çağrı / steady-state ayrımı — NumPy ve JAX CPU ölçümünde): NumPy'da
ölçümden önce 1 kez untimed warmup çalıştırılır (disk/cache ısıtma dışında
NumPy'nin ölçülecek bir "derleme" maliyeti yoktur). JAX'ta ise ilk çağrı hem
JIT derlemesini hem de gerçek çalışmayı içerir; bu çağrı warmup olarak
kullanılmaya devam eder, ama artık atılmıyor — süresi ayrıca "JAX first call
(compile + run)" olarak raporlanır. Sonrasındaki 10 tekrarlık "steady-state"
ölçümü, derlenmiş fonksiyonun saf çalışma süresini yansıtır. Hızlanma (speedup)
hesabı SADECE steady-state ortalamaları üzerinden yapılır; ilk çağrı süresi
speedup hesabına katılmaz.
"""

import statistics
import timeit

import numpy as np

import jax
import jax.numpy as jnp


N = 10_000_000
REPEATS = 10


# ---------------------------------------------------------------------------
# NumPy versiyonu (her zaman CPU'da, host belleğinde çalışır)
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
    print(f"  {label:<28}: {mean_ms:6.2f} ± {std_ms:5.2f} ms (n={len(times)})")


if __name__ == "__main__":
    print("=" * 62)
    print(f" DENEY 2: Kernel Fusion Kıyaslaması ({N:,} eleman)".replace(",", "."))
    print("=" * 62)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar: {list(devices.keys())}")

    # Veri BİR KEZ, host (NumPy) belleğinde üretilir; üretim süresi hiçbir
    # ölçüme dahil edilmez. float32: JAX'in varsayılan hassasiyetiyle eşleşsin
    # diye bilinçli olarak seçildi (bkz. dosya başındaki not).
    rng = np.random.default_rng(42)
    A_np = rng.random(N).astype(np.float32)
    B_np = rng.random(N).astype(np.float32)

    # --- Referans: NumPy (her zaman CPU/host'ta) ---
    numpy_chain(A_np, B_np)  # untimed warmup (cache/sayfa ısıtma; derleme yok)

    numpy_times, y_np = time_it(lambda: numpy_chain(A_np, B_np))
    numpy_mean = statistics.mean(numpy_times)

    print()
    print_stats("NumPy steady-state", numpy_times)

    cpu_mean = None
    gpu_compute_mean = None
    gpu_e2e_mean = None

    for name, device in devices.items():
        print(f"\n  --- {name} ---")

        if name != "GPU":
            # --- Tek cihaz ölçümü (CPU): veri bir kez cihaza taşınır,
            # sonrasında SADECE derlenmiş fonksiyonun çalışma süresi ölçülür. ---
            A_dev = jax.device_put(A_np, device)
            B_dev = jax.device_put(B_np, device)

            # İlk çağrı: JIT derlemesi + ilk çalıştırmayı birlikte içerir. Bu
            # çağrı warmup görevini de görür (sonraki çağrılar zaten derlenmiş
            # olacak), ama artık atılmıyor — süresi ayrıca raporlanıyor.
            first_call_time = timeit.timeit(lambda: jax_chain(A_dev, B_dev).block_until_ready(), number=1)
            print(f"  {'JAX first call (compile + run)':<28}: {first_call_time*1000:6.2f} ms")

            cpu_times, y_dev = time_it(lambda: jax_chain(A_dev, B_dev).block_until_ready())
            cpu_mean = statistics.mean(cpu_times)
            print_stats(f"JAX {name} steady-state", cpu_times)

            max_abs_diff = float(np.max(np.abs(y_np - np.asarray(y_dev))))
            print(f"  {'Maks. fark (sayısal kontrol)':<28}: {max_abs_diff:.2e}")
            continue

        # =====================================================================
        # GPU: iki ayrı, birbirine KARIŞTIRILMAMASI gereken ölçüm.
        # =====================================================================

        # --- 1) Compute-only: veri ÖNCEDEN GPU'ya taşınmış, warmup yapılmış.
        # Ölçülen SADECE: derlenmiş kernel'in GPU'daki çalışma süresi +
        # block_until_ready() ile senkronizasyon. Host<->GPU transferi YOK. ---
        A_gpu = jax.device_put(A_np, device)
        B_gpu = jax.device_put(B_np, device)

        jax_chain(A_gpu, B_gpu).block_until_ready()  # warmup (JIT derlemesi)

        gpu_compute_times, y_gpu = time_it(lambda: jax_chain(A_gpu, B_gpu).block_until_ready())
        gpu_compute_mean = statistics.mean(gpu_compute_times)
        print_stats("JAX GPU compute-only", gpu_compute_times)

        # --- 2) End-to-end (round-trip): her tekrarda host NumPy dizisinden
        # başlanır -> GPU'ya taşınır -> hesaplanır -> sonuç host'a geri getirilir.
        # jax.device_get() hem hesaplamanın bitmesini bekler (senkron) hem de
        # sonucu gerçekten host belleğine kopyalar. Fonksiyon zaten derlenmiş
        # olduğundan JIT derleme süresi bu ölçüme KARIŞMAZ. ---
        def round_trip():
            A_gpu_iter = jax.device_put(A_np, device)
            B_gpu_iter = jax.device_put(B_np, device)
            y_gpu_iter = jax_chain(A_gpu_iter, B_gpu_iter)
            return jax.device_get(y_gpu_iter)  # GPU -> host, NumPy dizisi olarak döner

        round_trip()  # warmup: allocator/transfer yollarını ısıt (JIT zaten hazır)
        gpu_e2e_times, y_gpu_host = time_it(round_trip)
        gpu_e2e_mean = statistics.mean(gpu_e2e_times)
        print_stats("JAX GPU end-to-end", gpu_e2e_times)

        max_abs_diff = float(np.max(np.abs(y_np - y_gpu_host)))
        print(f"  {'Maks. fark (sayısal kontrol)':<28}: {max_abs_diff:.2e}")

    print("\nHızlanma (NumPy CPU'ya göre, ortalama süreler üzerinden):")
    if cpu_mean is not None:
        print(f"  {'JAX CPU speedup':<28}: {numpy_mean / cpu_mean:.1f}x")
    if gpu_compute_mean is not None:
        print(f"  {'GPU compute-only speedup':<28}: {numpy_mean / gpu_compute_mean:.1f}x")
    if gpu_e2e_mean is not None:
        print(f"  {'GPU end-to-end speedup':<28}: {numpy_mean / gpu_e2e_mean:.1f}x")

    print("=" * 62)
