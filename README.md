# JAX Explained — Küçük ve Eğitici Deneyler

JAX'in temel özelliklerini (JIT derleme, Kernel Fusion, `vmap`, otomatik türev,
Pytree tabanlı durum yönetimi) klasik NumPy / saf Python yöntemleriyle
kıyaslayan, birbirinden bağımsız 7 küçük deney.

## Kurulum

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU'da çalıştırmak isterseniz:** `requirements.txt`'teki düz `jax` paketi pip'te
> varsayılan olarak CPU-only jaxlib kurar. Kodların hiçbiri cihazı CPU'ya sabitlemez;
> JAX görebildiği en iyi cihazı (GPU/TPU) otomatik kullanır — yeter ki uygun jaxlib
> kurulu olsun. CUDA'lı bir GPU için `requirements.txt` kurulumundan sonra (veya
> yerine) şunu çalıştırın:
> ```bash
> pip install -U "jax[cuda12]"
> ```

### Aynı makinede hem CPU hem GPU sonucu — tek çalıştırmada, otomatik

Ekstra bir ayar veya ortam değişkeni gerekmez: her deney dosyası çalışırken
kendi içinde `jax.devices("cpu")` ve `jax.devices("gpu")` ile bu makinede
neler bulunduğunu otomatik tespit eder ve JAX'e ait tüm bölümleri **her
bulunan cihazda ayrı ayrı** çalıştırıp sonuçları tek çıktıda yan yana basar
(`jax.device_put(...)` / `jax.default_device(...)` ile). Yani IDE'de sadece
▶️ **Run** tuşuna basmanız yeterli:

- Sadece CPU'lu bir makinede çalıştırırsanız çıktıda `Bulunan cihazlar: ['CPU']`
  görürsünüz, sonuçlar tek bölüm halinde gelir.
- CUDA'lı bir GPU'nun olduğu makinede çalıştırırsanız `Bulunan cihazlar: ['CPU', 'GPU']`
  görürsünüz ve her deney CPU ile GPU sonuçlarını art arda, aynı çalıştırmada
  gösterir — script'i tekrar çalıştırmanıza veya ortam değişkeni ayarlamanıza
  gerek kalmaz.

(İsterseniz yine de belirli bir backend'i zorlamak isteyebilirsiniz — örn.
GPU'lu bir makinede GPU'yu görmezden gelmek için `JAX_PLATFORMS=cpu python 01_loop_benchmark.py`
gibi. Ama günlük kullanım için bu gerekmez.)

## Çalıştırma

Her dosya tamamen bağımsızdır, tek başına çalıştırılabilir:

```bash
python 01_loop_benchmark.py
python 02_kernel_fusion.py
python 03_vmap_batching.py
python 04_autodiff_vs_numerical.py
python 05_mini_mlp_pytree.py
python 06_mnist_classifier.py
python 07_control_flow_primitives.py
```

## Deneyler

| Dosya | Konu | Kıyaslama |
|---|---|---|
| `01_loop_benchmark.py` | Döngüler | Python `for` vs `jax.lax.fori_loop` vs `jax.lax.scan` |
| `02_kernel_fusion.py` | Bellek / Kernel Fusion | NumPy vs `@jax.jit` |
| `03_vmap_batching.py` | Otomatik Vektörleştirme | Python döngüsü vs NumPy vs `jax.vmap` |
| `04_autodiff_vs_numerical.py` | Türev | NumPy Sonlu Farklar vs `jax.grad` / `jacfwd(jacrev(f))` |
| `05_mini_mlp_pytree.py` | Pytree & Fonksiyonel Eğitim | Saf `jax.grad` + `@jax.jit` ile tek eğitim adımı |
| `06_mnist_classifier.py` (bonus) | Uçtan uca eğitim | NumPy elle backprop vs `jax.grad` + `@jax.jit` ile MNIST sınıflandırma |
| `07_control_flow_primitives.py` | Kontrol Akışı | `lax.cond` / `lax.switch` / `lax.while_loop` vs Python `if`/`match`/`while`, + `vmap` entegrasyonu |

`06_mnist_classifier.py` ilk çalıştırmada MNIST verisini (~11 MB) internetten
indirip proje içindeki `.mnist_cache/` klasörüne kaydeder; sonraki
çalıştırmalarda diskten okur, internet gerekmez.

Not: İlk JAX çağrısı derleme (tracing/compilation) içerdiğinden, her ölçümden
önce bir "warmup" çağrısı yapılır ve zamanlama `.block_until_ready()` ile
asenkron çalıştırmayı bekleyerek doğru ölçülür.
