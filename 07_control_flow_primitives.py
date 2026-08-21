"""
Deney 7: JAX İleri Seviye Kontrol Akışı Primitifleri
=======================================================
Bu dosya, JAX'te "normal" Python kontrol akışının (if/else, while, döngü)
neden `@jax.jit` altında çoğu zaman ÇALIŞMADIĞINI ve bunun yerine JAX'in
kendi kontrol akışı primitiflerinin nasıl kullanıldığını gösterir:

  1) lax.cond        -> jit-uyumlu if/else (koşullu dallanma)
  2) lax.switch       -> jit-uyumlu match/case (çok yollu dallanma)
  3) lax.while_loop    -> jit-uyumlu, adım sayısı ÇALIŞMA ZAMANINDA belli olan döngü
  4) vmap + cond/switch -> bu primitiflerin bir batch üzerinde nasıl çalıştığı

Temel sebep: `@jax.jit` bir fonksiyonu SADECE BİR KEZ, soyut (traced) girdilerle
"izler" (trace) ve XLA için sabit bir hesaplama grafiği üretir. Bu grafik,
girdinin GERÇEK DEĞERİNE değil sadece şekil (shape) ve tipine (dtype) bakar.
Bu yüzden `if traced_x > 0:` gibi bir ifade, "traced_x hangi dalı seçeceğim
bilmiyorum" hatası verir — çünkü derleme anında traced_x'in gerçek değeri
henüz yoktur. lax.cond / lax.switch / lax.while_loop, dallanmayı Python
seviyesinde değil, XLA'nın kendi derlenmiş grafiğinin İÇİNDE ifade eder.
"""

import math
import time

import jax
import jax.numpy as jnp
from jax import lax

# lax.while_loop bölümünde tol=1e-9 gibi sıkı bir yakınsama eşiği kullanıyoruz.
# JAX'in VARSAYILAN float32 hassasiyeti (~1e-7) bu eşiğe hiç ULAŞAMAZ ve
# cond_fun sonsuza kadar True kalıp SONSUZ DÖNGÜYE girer. Bu yüzden x64
# hassasiyetini (float64) açıyoruz — deney 4'teki ile aynı gereklilik.
jax.config.update("jax_enable_x64", True)


# ===========================================================================
# 1) lax.cond vs Python if/else
# ===========================================================================
def leaky_relu_python(x):
    # Bu fonksiyon SADECE x somut (concrete) bir Python/NumPy sayısıyken çalışır.
    # jit altında x bir "tracer" (soyut iz) olduğundan `if x > 0` çalışamaz.
    if x > 0:
        return x
    else:
        return 0.01 * x


def leaky_relu_cond(x):
    # lax.cond(pred, true_fn, false_fn, operand)
    # KRİTİK KURAL: true_fn ve false_fn'in döndürdüğü değerler AYNI shape ve
    # AYNI dtype'a sahip olmak ZORUNDADIR. XLA hangi dalın seçileceğini
    # derleme zamanında bilemediği için HER İKİ dalı da derler; bu yüzden
    # dallar arasında "şekil uyuşmazlığı" olamaz.
    return lax.cond(x > 0, lambda v: v, lambda v: 0.01 * v, x)


def demo_cond():
    print("\n" + "-" * 62)
    print("  [1] lax.cond vs Python if/else")
    print("-" * 62)

    x = 3.5
    print(f"  Python if/else, jit'siz çağrı (x={x}) -> {leaky_relu_python(x)}  (sorunsuz)")

    # Aynı Python fonksiyonunu jit edip traced bir girdiyle çağırmayı DENE:
    try:
        jax.jit(leaky_relu_python)(jnp.array(x))
        print("  jit(Python if/else) -> beklenmedik şekilde çalıştı (olmamalıydı!)")
    except Exception as e:
        first_line = str(e).splitlines()[0]
        print(f"  jit(Python if/else) -> HATA: {type(e).__name__}")
        print(f"    -> {first_line}")

    # lax.cond ile aynı mantık, jit altında sorunsuz çalışır:
    jitted_cond = jax.jit(leaky_relu_cond)
    _ = jitted_cond(jnp.array(1.0)).block_until_ready()  # warmup (derleme)

    print("\n  jit(lax.cond) sonuçları:")
    for test_x in [3.5, -2.0, 0.0]:
        result = jitted_cond(jnp.array(test_x)).block_until_ready()
        print(f"    leaky_relu_cond(x={test_x:>5}) -> {float(result):.4f}")


# ===========================================================================
# 2) lax.switch: dinamik, çok yollu dallanma (match/case karşılığı)
# ===========================================================================
def linear(x):
    return x


def relu(x):
    return jnp.maximum(x, 0.0)


def square(x):
    return x ** 2


def sigmoid(x):
    return jax.nn.sigmoid(x)


ACTIVATIONS = [linear, relu, square, sigmoid]
ACTIVATION_NAMES = ["linear", "relu", "square", "sigmoid"]


def apply_activation_switch(index, x):
    # lax.switch(index, branches, operand)
    # KRİTİK KURAL 1: `branches` listesindeki HER fonksiyon aynı shape/dtype
    # döndürmelidir (lax.cond ile aynı kısıt, burada N dal için geçerli).
    # KRİTİK KURAL 2: `index` sınırların dışındaysa (örn. -1 ya da 99),
    # JAX hata vermez; index otomatik olarak [0, len(branches)-1] aralığına
    # KIRPILIR (clamp). Bu, NumPy/Python indeksleme davranışından farklıdır!
    return lax.switch(index, ACTIVATIONS, x)


def apply_activation_python(index, x):
    # Karşılaştırma için: aynı mantığın saf Python match/case karşılığı.
    match index:
        case 0:
            return x
        case 1:
            return max(x, 0.0)
        case 2:
            return x ** 2
        case 3:
            return 1.0 / (1.0 + math.exp(-x))
        case _:
            raise ValueError(f"Geçersiz index: {index}")


def demo_switch():
    print("\n" + "-" * 62)
    print("  [2] lax.switch (dinamik çok yollu dallanma)")
    print("-" * 62)

    jitted_switch = jax.jit(apply_activation_switch)
    _ = jitted_switch(0, jnp.array(1.0)).block_until_ready()  # warmup

    x = 2.0
    print(f"  Girdi x = {x}\n")
    print(f"  {'index':<7}{'fonksiyon':<12}{'lax.switch':<14}{'Python match/case':<20}{'fark'}")
    for idx in range(4):
        jax_result = float(jitted_switch(idx, jnp.array(x)).block_until_ready())
        py_result = apply_activation_python(idx, x)
        diff = abs(jax_result - py_result)
        print(f"  {idx:<7}{ACTIVATION_NAMES[idx]:<12}{jax_result:<14.4f}{py_result:<20.4f}{diff:.2e}")

    # Kırpma (clamp) davranışını göster: aralık dışı bir index verelim.
    out_of_range_result = float(jitted_switch(99, jnp.array(x)).block_until_ready())
    print(f"\n  index=99 (aralık dışı) -> son dala kırpılır -> sigmoid({x}) = {out_of_range_result:.4f}")


# ===========================================================================
# 3) lax.while_loop: adım sayısı ÇALIŞMA ZAMANINDA belli olan dinamik döngü
# ===========================================================================
def sqrt_python_while(target: float, tol: float = 1e-9, guess: float = 1.0):
    # Referans: Newton-Raphson yöntemiyle saf Python'da karekök bulma.
    x = guess
    iters = 0
    while abs(x * x - target) > tol:
        x = 0.5 * (x + target / x)
        iters += 1
    return x, iters


def sqrt_jax_while(target, tol=1e-9, guess=1.0):
    # lax.while_loop(cond_fun, body_fun, init_val)
    # KRİTİK KURAL: lax.scan/fori_loop'un aksine burada adım sayısı SABİT
    # DEĞİLDİR; koşul (cond_fun) her adımda çalışma zamanında değerlendirilir.
    # Bu esneklik bir bedelle gelir: lax.while_loop üzerinden REVERSE-MODE
    # otomatik türev (jax.grad) ALINAMAZ (XLA, geriye yayılım için sabit
    # bellek/adım sayısı gerektirir). Türev gerekiyorsa lax.fori_loop veya
    # sabit adım sayılı lax.scan tercih edilmelidir.
    def cond_fun(state):
        x, i = state
        return jnp.abs(x * x - target) > tol

    def body_fun(state):
        x, i = state
        x_new = 0.5 * (x + target / x)
        return (x_new, i + 1)

    final_x, final_i = lax.while_loop(cond_fun, body_fun, (guess, 0))
    return final_x, final_i


def demo_while_loop():
    print("\n" + "-" * 62)
    print("  [3] lax.while_loop (dinamik yakınsama: Newton-Raphson karekök)")
    print("-" * 62)

    target = 2.0

    py_result, py_iters = sqrt_python_while(target)

    jitted_while = jax.jit(sqrt_jax_while)
    jax.block_until_ready(jitted_while(1.0))  # warmup

    jax_x, jax_iters = jitted_while(float(target))
    jax.block_until_ready((jax_x, jax_iters))

    print(f"  Hedef: sqrt({target})")
    print(f"    Python while  -> sonuç={py_result:.10f}  adım={py_iters}")
    print(f"    lax.while_loop -> sonuç={float(jax_x):.10f}  adım={int(jax_iters)}")
    print(f"    Gerçek değer  -> {math.sqrt(target):.10f}")

    # lax.while_loop üzerinden türev almayı DENE -> hata bekleniyor.
    print("\n  jax.grad(lax.while_loop tabanlı fonksiyon) deneniyor...")
    try:
        grad_fn = jax.grad(lambda t: sqrt_jax_while(t)[0])
        grad_fn(target)
        print("  -> beklenmedik şekilde çalıştı (olmamalıydı!)")
    except Exception as e:
        print(f"  -> HATA (beklenen davranış): {type(e).__name__}")
        print(f"     -> {str(e).splitlines()[0]}")


# ===========================================================================
# 4) vmap + lax.cond / lax.switch: kontrol akışını batch'e taşımak
# ===========================================================================
def demo_vmap_control_flow():
    print("\n" + "-" * 62)
    print("  [4] vmap + lax.cond / lax.switch (batch üzerinde paralel çalışma)")
    print("-" * 62)

    # --- vmap + lax.cond ---
    # KRİTİK KURAL: vmap altında lax.cond GERÇEK dallanma yapmaz! XLA bunu
    # elementwise bir "select" işlemine çevirir: batch'in HER elemanı için
    # HEM true_fn HEM false_fn hesaplanır, sonra doğru sonuç seçilir.
    # Yani vmap+cond hesaplama TASARRUFU sağlamaz (pahalı bir dal atlanmaz);
    # sağladığı şey SIMD tarzı, tek bir derlenmiş grafikle toplu doğru sonuçtur.
    batched_leaky_relu = jax.jit(jax.vmap(leaky_relu_cond))
    xs = jnp.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    _ = batched_leaky_relu(xs).block_until_ready()  # warmup
    cond_results = batched_leaky_relu(xs).block_until_ready()

    print("  vmap(lax.cond) ile toplu leaky-ReLU:")
    for x_val, y_val in zip(xs.tolist(), cond_results.tolist()):
        print(f"    x={x_val:>5.1f} -> {y_val:.4f}")

    # --- vmap + lax.switch ---
    # Aynı kural lax.switch için de geçerli: batch'in HER elemanı için
    # TÜM dallar hesaplanır, sonra her eleman kendi index'ine göre seçim yapar.
    batched_switch = jax.jit(jax.vmap(apply_activation_switch, in_axes=(0, 0)))
    indices = jnp.array([0, 1, 2, 3, 1])
    xs2 = jnp.array([-2.0, -2.0, -2.0, -2.0, 2.0])
    _ = batched_switch(indices, xs2).block_until_ready()  # warmup
    switch_results = batched_switch(indices, xs2).block_until_ready()

    print("\n  vmap(lax.switch) ile toplu, indeks-bazlı aktivasyon seçimi:")
    for idx, x_val, y_val in zip(indices.tolist(), xs2.tolist(), switch_results.tolist()):
        print(f"    index={idx} ({ACTIVATION_NAMES[idx]:<7}) x={x_val:>5.1f} -> {y_val:.4f}")


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 7: JAX İleri Seviye Kontrol Akışı Primitifleri")
    print("=" * 62)
    print(f"  Kullanılan cihaz(lar): {jax.devices()}")

    demo_cond()
    demo_switch()
    demo_while_loop()
    demo_vmap_control_flow()

    print("\n" + "=" * 62)
    print(" Özet:")
    print("  lax.cond        -> jit-uyumlu if/else (dallar aynı shape/dtype)")
    print("  lax.switch      -> jit-uyumlu match/case (index otomatik kırpılır)")
    print("  lax.while_loop  -> dinamik adım sayısı, ama reverse-mode grad YOK")
    print("  vmap + cond/switch -> TÜM dallar batch'te hesaplanır, sonra seçilir")
    print("=" * 62)
