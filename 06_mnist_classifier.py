"""
Deney 6: NumPy vs PyTorch (eager) vs PyTorch (compiled) vs Flax NNX + Optax
=============================================================================
Aynı MNIST MLP'sini (784 -> 128 (tanh) -> 10) DÖRT farklı şekilde eğitiyoruz:

  1) NumPy + ELLE türetilmiş backpropagation
     Model/katman = elle dict; backward = elle zincir kuralı; optimizer = elle
     `p - lr*g`.

  2) PyTorch (eager) — nn.Module + autograd + torch.optim.SGD
     Model/katman = `torch.nn`; backward = `autograd` (`loss.backward()`);
     optimizer = `torch.optim.SGD`. Derleme YOK, her adım doğrudan (eager)
     yorumlanarak çalışır.

  3) PyTorch (compiled) — aynı model/optimizer, `torch.compile` ile derlenmiş
     Yalnızca training step `torch.compile` ile sarmalanır; geri kalan HER ŞEY
     (2) ile birebir aynıdır. Platform/sürüm desteklemiyorsa bu bölüm
     ZARARSIZCA atlanır (aşağıya bakınız).

  4) Flax NNX + Optax + JAX
     Model/katman = Flax NNX (`nnx.Module`, `nnx.Linear`); backward =
     JAX/NNX autodiff (`nnx.value_and_grad`); optimizer = Optax
     (`optax.sgd` + `nnx.Optimizer`); derleme = NNX/JAX JIT (`@nnx.jit`).

PEDAGOJİK AMAÇ: Bu deneyin amacı hangi yöntemin "daha hızlı" olduğunu
kanıtlamak DEĞİLDİR. Asıl fayda; aynı sinir ağını dört farklı abstraction
seviyesinde yazarak her katmanın (autograd, optimizer, derleme) NE KADAR
boilerplate ortadan kaldırdığını somut kodla göstermektir:

    NumPy:    model+forward+backward+SGD TAMAMEN elle
    PyTorch:  model/layer -> torch.nn ; backward -> autograd ; optimizer -> torch.optim
    Flax/JAX: model/layer -> Flax NNX ; backward -> JAX/NNX autodiff ;
              optimizer -> Optax ; derleme -> JAX/NNX JIT

Performans sonuçları OLDUĞU GİBİ raporlanır; herhangi bir yöntem NumPy'dan
CPU'da yavaş çıkarsa bu SAKLANMAZ veya DEĞİŞTİRİLMEZ (aşağıdaki "Not"
bloğuna bakınız — sebebi genelde per-batch Python dispatch maliyetidir).

## Ortak deney ayarları (DÖRT yöntem için de BİREBİR AYNI)
  - 10.000 eğitim / 2.000 test örneği, float32
  - Mimari: 784 -> 128 (tanh) -> 10
  - 5 epoch, batch boyutu 100, SGD (momentumsuz), öğrenme oranı 0.5
  - AYNI batch sırası (`build_epoch_indices`, sabit tohum)
  - MÜMKÜN OLDUĞUNCA aynı başlangıç ağırlıkları: `init_params()` ile HOST'ta
    (NumPy) üretilen W1/b1/W2/b2, hem PyTorch hem Flax NNX modelinin kendi
    rastgele ilklendirmesinin ÜZERİNE doğrudan yazılarak atanıyor.
    ÖNEMLİ (PyTorch'a özgü ayrıntı): `torch.nn.Linear` ağırlıkları
    `(out_features, in_features)` şeklinde tutar — bizim `(in, out)`
    kuralımızın TERSİ. Bu yüzden `W1`/`W2` TRANSPOZE edilerek atanır;
    `x @ W1 + b1` ile `x @ weight.T + bias` (weight = W1.T) matematiksel
    olarak AYNI şeydir.
  - Aynı loss matematiği: softmax + cross-entropy (integer label), batch
    ortalaması. NumPy elle (one-hot), PyTorch `nn.CrossEntropyLoss`, Flax/JAX
    `optax.softmax_cross_entropy_with_integer_labels` — üçü de matematiksel
    olarak eşdeğerdir.

## Ölçüm metodolojisi (PyTorch ve Flax/JAX için AYNI)
  - Önce TEK BİR sanity check (ölçüm dışı): reset sonrası ağırlıkların
    GERÇEKTEN NumPy init ile aynı olduğu doğrulanır.
  - Ardından, HENÜZ hiçbir training-step/accuracy çağrısı yapılmamışken,
    pristine durumdan "first training run" ölçülür (`verbose=False`,
    repeat=1) — bu GERÇEKTEN derleme/graph-capture + tam 5 epoch'u birlikte
    içerir (PyTorch compiled: `torch.compile` graph capture; Flax/JAX:
    `@nnx.jit` derlemesi).
  - Sonra sıfırlanıp UNTIMED bir `verbose=True` "gösterim" koşusu yapılır
    (artık derleme/graph zaten sıcaktır; SADECE epoch ilerlemesini gösterir,
    ölçüme dahil DEĞİLDİR). Bu koşunun epoch 5 sonundaki doğruluğu, aynı
    pristine durumdan başladığı için "first training run" doğruluğuyla
    (determinizm gereği) BİREBİR AYNI olmalıdır.
  - Sonra, her tekrar öncesi yine pristine duruma sıfırlanarak 10x
    steady-state ölçülür; ortalama ± standart sapma raporlanır.
  - Reset, HOST transferi GEREKTİRMEZ: cihazda tutulan BAĞIMSIZ bir kopya
    (PyTorch: `state_dict()` klonu; Flax: `nnx.clone(nnx.state(...))`) geri
    yüklenir. Basit/momentumsuz SGD'de optimizer'ın kendisi DURUMSUZDUR, bu
    yüzden optimizer'ı yeniden kurmaya gerek yoktur.
  - GPU varsa (PyTorch ve Flax/JAX için AYRI AYRI) iki sonuç raporlanır:
      * GPU compute-only : veri + model durumu + batch indeksleri ÖNCEDEN
        GPU'da; döngü içinde HİÇBİR host<->device transferi/senkronizasyonu
        YOK (epoch kaybı device üzerinde biriktirilir, host'a çekme SADECE
        verbose modda).
      * GPU end-to-end    : her tekrarda host verisinden/ağırlıklarından
        başlanır, GPU'ya taşınır, 5 epoch eğitilir, son doğruluk host'a geri
        getirilir (tam round-trip).
    Bu iki sonuç birbirine KARIŞTIRILMAZ.
  - ÖNEMLİ (torch.compile'a özgü): model, optimizer ve derlenmiş training-step
    SARMALAYICISI tekrarlar boyunca YENİDEN OLUŞTURULMAZ; aynı nesneler yeniden
    kullanılır. `torch.compile` guard'ları `model`/`optimizer` nesne kimliğine
    (`check_obj_id`) bakar; her round-trip'te yeni bir optimizer üretilirse guard
    düşer, Dynamo aynı frame'i tekrar tekrar derler ve `recompile_limit` aşılınca
    eager fallback'e düşerek "compiled" ölçümünü SESSİZCE kirletir. Bunun yerine
    end-to-end tekrarında yalnızca (i) veri host'tan cihaza kopyalanır ve (ii)
    pristine ağırlıklar HOST'tan `load_state_dict` ile MEVCUT cihaz modeline
    yüklenir (gerçek host->device transferi; `load_state_dict` var olan tensörlere
    `.copy_()` yapar, yeni parametre nesnesi OLUŞTURMAZ). Böylece round-trip
    semantiği AYNEN korunur ama yeniden derleme tetiklenmez. Doğrulamak için:
        TORCH_LOGS=recompiles python 06_mnist_classifier.py
    (Uyarıyı `torch._dynamo.config.recompile_limit` ile gizlemek bir çözüm
    DEĞİLDİR; asıl guard sebebi ortadan kaldırılmıştır.)
  - Veri yükleme (MNIST indirme), ön işleme ve rastgele ağırlık üretimi
    hiçbir ölçüme dahil DEĞİLDİR.
  - NOT: PyTorch'un GPU tespiti (`torch.cuda.is_available()`) ile JAX'in GPU
    tespiti BİRBİRİNDEN BAĞIMSIZDIR (farklı kütüphaneler, farklı backend
    derlemeleri); bu yüzden biri GPU görüp diğeri görmeyebilir. Çıktıda
    "PyTorch GPU" ve "Flax/JAX GPU" ayrı ayrı etiketlenir.

Not (TF32): CUDA GPU'larında PyTorch, float32 matmul'ler için TF32 tensor
core'larının "mevcut ama etkin değil" olduğunu söyleyen bir uyarı basabilir. Bu
bir hata/correctness uyarısı değil, performans önerisidir. NumPy/JAX ile adil
kıyas bozulmasın diye `torch.set_float32_matmul_precision` ÇAĞRILMAZ; PyTorch'un
varsayılan float32 matmul hassasiyeti KORUNUR.

Not: İlk çalıştırmada MNIST verisi (~11 MB) internetten indirilip
`.mnist_cache/` klasörüne kaydedilir; sonraki çalıştırmalarda diskten okunur.

ÖNEMLİ: Bu betiği çalıştırmak için `flax`, `optax` ve `torch` kurulu olmalı:
    pip install flax optax torch
`torch.compile` (PyTorch >= 2.0 gerektirir) bu ortamda/platformda
desteklenmiyorsa "PyTorch (compiled)" bölümü otomatik olarak, hata
vermeden atlanır ve bu açıkça raporlanır.
"""

import os
import statistics
import timeit
import urllib.request

import numpy as np

import jax
import jax.numpy as jnp
import optax
import torch
from flax import nnx


# ---------------------------------------------------------------------------
# Ayarlar (DÖRT yöntem için de AYNI)
# ---------------------------------------------------------------------------
N_TRAIN = 10_000
N_TEST = 2_000
HIDDEN_DIM = 128
BATCH_SIZE = 100
EPOCHS = 5
LEARNING_RATE = 0.5
REPEATS = 10

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mnist_cache", "mnist.npz")


# ---------------------------------------------------------------------------
# Veri: indir (bir kere) + önbellekten oku + ön işle
# ---------------------------------------------------------------------------
def load_mnist():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if not os.path.exists(CACHE_PATH):
        print(f"  MNIST verisi indiriliyor (~11 MB)...\n  Kaynak: {MNIST_URL}")
        urllib.request.urlretrieve(MNIST_URL, CACHE_PATH)
        print("  İndirme tamamlandı, önbelleğe kaydedildi.")
    else:
        print(f"  MNIST verisi önbellekten okunuyor: {CACHE_PATH}")

    with np.load(CACHE_PATH) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
    return (x_train, y_train), (x_test, y_test)


def preprocess(x, y, n_samples, rng):
    idx = rng.choice(len(x), size=n_samples, replace=False)
    x = x[idx].reshape(n_samples, -1).astype(np.float32) / 255.0  # düzleştir + normalize et
    y_labels = y[idx].astype(np.int32)
    y_onehot = np.zeros((n_samples, 10), dtype=np.float32)
    y_onehot[np.arange(n_samples), y_labels] = 1.0
    return x, y_labels, y_onehot


# ---------------------------------------------------------------------------
# Ortak ağırlık ilklendirme (DÖRT yöntem de AYNI başlangıç noktasından başlar)
# ---------------------------------------------------------------------------
def init_params(rng):
    return {
        "W1": (rng.standard_normal((784, HIDDEN_DIM)) * np.sqrt(1.0 / 784)).astype(np.float32),
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W2": (rng.standard_normal((HIDDEN_DIM, 10)) * np.sqrt(1.0 / HIDDEN_DIM)).astype(np.float32),
        "b2": np.zeros(10, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Host tarafında, SABİT tohumla epoch/batch indekslerini ÖNCEDEN üretir; her
# yöntem de TAM OLARAK aynı batch sırasını kullanır.
# Dönüş: (EPOCHS, n_batches, BATCH_SIZE) NumPy int32 dizisi.
# ---------------------------------------------------------------------------
def build_epoch_indices(n_samples, batch_size, epochs, seed=1):
    rng = np.random.default_rng(seed)
    n_batches = n_samples // batch_size
    idx = np.empty((epochs, n_batches, batch_size), dtype=np.int32)
    for e in range(epochs):
        perm = rng.permutation(n_samples)
        for b in range(n_batches):
            idx[e, b] = perm[b * batch_size:(b + 1) * batch_size]
    return idx


# ===========================================================================
# 1) NUMPY: elle türetilmiş backpropagation
#
#    ELLE YAPILANLAR (bu yöntemin tamamı):
#      - Katman parametrelerinin organizasyonu (W1/b1/W2/b2 dict'i, elle taşınır)
#      - İleri yayılım (forward_np)
#      - Softmax + Cross-Entropy hesaplama (softmax_np, loss_np)
#      - Geri yayılım / zincir kuralı: dz2, dW2, db2, dh, dz1, dW1, db1
#      - SGD güncellemesi: her ağırlık için elle `w - lr * dw`
# ===========================================================================
def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward_np(params, x):
    z1 = x @ params["W1"] + params["b1"]
    h = np.tanh(z1)
    z2 = h @ params["W2"] + params["b2"]
    return z1, h, z2


def loss_np(params, x, y_onehot):
    _, _, z2 = forward_np(params, x)
    probs = softmax_np(z2)
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))


def train_step_np(params, x, y_onehot, lr):
    n = x.shape[0]
    z1, h, z2 = forward_np(params, x)
    probs = softmax_np(z2)

    # -- elle türetilmiş zincir kuralı (chain rule) --
    dz2 = (probs - y_onehot) / n
    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0)

    dh = dz2 @ params["W2"].T
    dz1 = dh * (1 - np.tanh(z1) ** 2)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)

    # -- elle SGD güncellemesi: p - lr * g --
    new_params = {
        "W1": params["W1"] - lr * dW1,
        "b1": params["b1"] - lr * db1,
        "W2": params["W2"] - lr * dW2,
        "b2": params["b2"] - lr * db2,
    }
    loss = loss_np(params, x, y_onehot)
    return new_params, loss


def accuracy_np(params, x, y_labels):
    _, _, z2 = forward_np(params, x)
    preds = z2.argmax(axis=1)
    return float(np.mean(preds == y_labels))


def run_numpy_training(init, x_train, y_train_onehot, x_test, y_test_labels, idx_array, verbose=False):
    params = {k: v.copy() for k, v in init.items()}
    epochs, n_batches, _ = idx_array.shape
    for epoch in range(epochs):
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = idx_array[epoch, b]
            params, batch_loss = train_step_np(params, x_train[idx], y_train_onehot[idx], LEARNING_RATE)
            epoch_loss += batch_loss
        if verbose:
            test_acc = accuracy_np(params, x_test, y_test_labels)
            print(f"  [NumPy] Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/n_batches:.4f}  test_acc={test_acc*100:.2f}%")
    return accuracy_np(params, x_test, y_test_labels)


# ===========================================================================
# 2) PYTORCH (eager) VE 3) PYTORCH (compiled)
#
#    NE ELLE YAZILMIYOR (kaldırılan boilerplate) VE KİM SAĞLIYOR:
#      - Model/layer organizasyonu -> torch.nn (nn.Module, nn.Linear)
#      - Geri yayılım (backward pass) -> autograd (`loss.backward()`)
#      - Optimizer / update mantığı -> torch.optim.SGD
#      - (Sadece "compiled" varyantında) training step derlemesi -> torch.compile
# ===========================================================================
class TorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, HIDDEN_DIM)
        self.fc2 = torch.nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


_torch_criterion = torch.nn.CrossEntropyLoss()  # softmax + NLL, batch ortalaması (integer label)


def load_init_into_torch(model, init_host):
    """`torch.nn.Linear` ağırlıkları (out_features, in_features) şeklinde
    tutulur — bizim (in, out) kuralımızın TERSİ. `weight = W.T` atayarak
    `x @ weight.T + bias`'ı `x @ W + b` ile matematiksel olarak eşitliyoruz."""
    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(init_host["W1"].T.copy()))
        model.fc1.bias.copy_(torch.from_numpy(init_host["b1"].copy()))
        model.fc2.weight.copy_(torch.from_numpy(init_host["W2"].T.copy()))
        model.fc2.bias.copy_(torch.from_numpy(init_host["b2"].copy()))


def train_step_torch(model, optimizer, x, y_labels):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = _torch_criterion(logits, y_labels)
    loss.backward()  # backward pass OTOMATİK (NumPy'daki dz2/dW2/... YOK)
    optimizer.step()  # SGD update OTOMATİK (NumPy'daki "p - lr*g" YOK)
    return loss.detach()  # host'a senkron OLMAZ


def accuracy_torch(model, x, y_labels):
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)
        return (preds == y_labels).float().mean()


def run_torch_epochs(model, optimizer, step_fn, x_train_dev, y_train_labels_dev, x_test_dev, y_test_labels_dev, idx_dev, label="", verbose=False):
    """SADECE hesaplama: tüm girdiler ZATEN cihazda kabul edilir. `verbose=False`
    iken döngü içinde HİÇBİR `.item()`/host senkronizasyonu YAPILMAZ (epoch
    kaybı device üzerinde tensör olarak biriktirilir)."""
    epochs, n_batches, _ = idx_dev.shape
    for epoch in range(epochs):
        epoch_loss = torch.zeros((), device=x_train_dev.device)
        for b in range(n_batches):
            batch_idx = idx_dev[epoch, b]
            loss = step_fn(model, optimizer, x_train_dev[batch_idx], y_train_labels_dev[batch_idx])
            epoch_loss = epoch_loss + loss  # device üzerinde kalır, host'a senkron OLMAZ
        if verbose:
            test_acc = accuracy_torch(model, x_test_dev, y_test_labels_dev).item()
            print(f"  [{label}] Epoch {epoch+1}/{EPOCHS}  loss={(epoch_loss/n_batches).item():.4f}  test_acc={test_acc*100:.2f}%")
    return model, optimizer


def torch_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_torch_and_sync(model, optimizer, step_fn, *args, **kwargs):
    device = next(model.parameters()).device
    model, optimizer = run_torch_epochs(model, optimizer, step_fn, *args, **kwargs)
    torch_sync(device)  # GERÇEKTEN cihaz hesaplamasının bittiğinden emin ol
    return model, optimizer


def reset_torch_optimizer_state(optimizer):
    """Optimizer'ı pristine duruma getirir — ama optimizer NESNESİNİ DEĞİŞTİRMEZ.

    `torch.compile` guard'ları optimizer'ın object identity'sine (`check_obj_id`)
    bakar; her tekrarda YENİ bir optimizer oluşturmak guard'ı düşürür ve yeniden
    derleme tetikler. Momentumsuz SGD zaten DURUMSUZDUR (per-parametre state
    tutmaz); yine de "pristine" garantisi için gradyanlar ve (varsa) per-parametre
    state AYNI nesneler üzerinde temizlenir."""
    optimizer.zero_grad(set_to_none=True)
    if optimizer.state:
        optimizer.state.clear()  # AYNI dict nesnesi korunur (yeniden atama YOK)


def prepare_torch(device, init_host, compiled=False, seed=0):
    """Modeli/optimizer'ı/derlenmiş step'i BİR KEZ kurar ve iki ayrı reset yolu döndürür.

    Bu fonksiyon her cihaz için YALNIZCA BİR KEZ çağrılır; döndürülen `model`,
    `optimizer` ve `step_fn` (derlenmiş sarmalayıcı) TÜM ölçüm tekrarları boyunca
    YENİDEN KULLANILIR — aksi halde `torch.compile` guard'ları her tekrarda düşer
    ve `recompile_limit` aşılır (bkz. dosya başındaki not).

    Dönüş:
      reset_fn()           -> compute-only reset: cihazda tutulan BAĞIMSIZ pristine
                              kopyadan geri yükler, HOST transferi YOKTUR.
      step_fn              -> training step (compiled ise `torch.compile` sarmalayıcısı).
      load_from_host_fn()  -> end-to-end reset: pristine ağırlıkları HOST'tan mevcut
                              cihaz modeline yükler (GERÇEK host->device transferi).

    Her iki reset de `load_state_dict` kullanır; bu VAR OLAN parametre tensörlerine
    `.copy_()` yapar (yeni nesne OLUŞTURMAZ), bu yüzden derleme önbelleği BOZULMAZ.
    Momentumsuz SGD durumsuz olduğundan optimizer'ı yeniden kurmaya gerek yoktur."""
    torch.manual_seed(seed)
    model = TorchMLP().to(device)
    load_init_into_torch(model, init_host)  # NumPy init ile BİREBİR AYNI başlangıç
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)  # momentum=0: elle "p - lr*g" ile MATEMATİKSEL olarak AYNI

    step_fn = torch.compile(train_step_torch) if compiled else train_step_torch

    pristine_state = {k: v.clone().detach() for k, v in model.state_dict().items()}  # cihazda, BAĞIMSIZ kopya
    host_pristine_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}  # HOST'ta, BAĞIMSIZ kopya

    def reset_fn():
        with torch.no_grad():
            model.load_state_dict(pristine_state)  # cihaz-içi reset, host transferi YOK
        reset_torch_optimizer_state(optimizer)
        return model, optimizer

    def load_from_host_fn():
        with torch.no_grad():
            model.load_state_dict(host_pristine_state)  # HOST -> cihaz kopyası (round-trip'in parçası)
        reset_torch_optimizer_state(optimizer)
        return model, optimizer

    return reset_fn, step_fn, load_from_host_fn


def print_tf32_note():
    """TF32 uyarısı bir performans önerisidir; adil kıyas için AYAR DEĞİŞTİRMİYORUZ."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("  Not: GPU TF32 tensor core'larını destekliyor, ancak NumPy/JAX ile adil kıyas")
        print("       için PyTorch'un varsayılan float32 matmul precision'ı KORUNUYOR")
        print("       (`torch.set_float32_matmul_precision` çağrılmıyor).")


def get_available_torch_devices() -> dict:
    devices = {"CPU": torch.device("cpu")}
    if torch.cuda.is_available():
        devices["GPU"] = torch.device("cuda")
    return devices


def check_torch_matches_init(model, init_host, tag):
    with torch.no_grad():
        ok = (
            torch.allclose(model.fc1.weight.detach().cpu(), torch.from_numpy(init_host["W1"].T.copy()))
            and torch.allclose(model.fc1.bias.detach().cpu(), torch.from_numpy(init_host["b1"].copy()))
            and torch.allclose(model.fc2.weight.detach().cpu(), torch.from_numpy(init_host["W2"].T.copy()))
            and torch.allclose(model.fc2.bias.detach().cpu(), torch.from_numpy(init_host["b2"].copy()))
        )
    assert ok, f"[{tag}] pristine ağırlıklar NumPy init ile eşleşmiyor!"
    print(f"  [{tag}] Sanity check: reset sonrası ağırlıklar NumPy init ile eşleşiyor. ✓")


def benchmark_torch_implementation(label, compiled, devices_torch, init_host, x_train, y_train_labels, x_test, y_test_labels, idx_array):
    """PyTorch (eager) VE PyTorch (compiled) için ORTAK ölçüm iskeleti — Flax
    NNX ile AYNI metodoloji: sanity check -> first training run (derleme
    dahil, henüz hiçbir step/accuracy çağrısı yapılmadan) -> untimed demo ->
    10x steady-state. GPU'da compute-only + end-to-end AYRI raporlanır.
    Dönüş: {'CPU'|'GPU compute-only': (first_run, times, acc), 'GPU end-to-end': (times, acc)}."""
    results = {}

    for name, device in devices_torch.items():
        print("\n" + "-" * 62)
        print(f"  PyTorch ({label}) — {name}")
        print("-" * 62)

        x_train_dev = torch.from_numpy(x_train).to(device)
        y_train_labels_dev = torch.from_numpy(y_train_labels.astype(np.int64)).to(device)
        x_test_dev = torch.from_numpy(x_test).to(device)
        y_test_labels_dev = torch.from_numpy(y_test_labels.astype(np.int64)).to(device)
        idx_dev = torch.from_numpy(idx_array.astype(np.int64)).to(device)

        # Model / optimizer / derlenmiş step BİR KEZ kurulur ve tüm ölçüm
        # tekrarlarında (compute-only VE end-to-end) YENİDEN KULLANILIR.
        reset_fn, step_fn, load_from_host_fn = prepare_torch(device, init_host, compiled=compiled)

        # --- Sanity check (TEK sefer, ölçüm dışı) ---
        sanity_model, _ = reset_fn()
        check_torch_matches_init(sanity_model, init_host, f"PyTorch {label}/{name}")

        # (a) "first training run": PRISTINE durumdan, verbose=False, HENÜZ
        # hiçbir step_fn/accuracy_torch çağrısı yapılmamışken ölçülür.
        # `compiled=True` iken bu GERÇEKTEN torch.compile graph-capture'ı +
        # tam 5 epoch'u birlikte içerir. Platform/sürüm `torch.compile`'ı
        # desteklemiyorsa burada hata fırlar; bu durumu zararsızca yakalayıp
        # bu cihaz için "PyTorch (compiled)"i atlıyoruz.
        try:
            first_run_times, (first_model, first_optimizer) = time_it(
                lambda: run_torch_and_sync(
                    *reset_fn(), step_fn, x_train_dev, y_train_labels_dev, x_test_dev, y_test_labels_dev, idx_dev,
                ),
                repeat=1,
            )
        except Exception as e:
            print(f"  PyTorch ({label}) [{name}] bu ortamda/platformda çalıştırılamadı, atlanıyor.\n    -> {type(e).__name__}: {e}")
            continue

        first_run_time = first_run_times[0]
        display_acc = accuracy_torch(first_model, x_test_dev, y_test_labels_dev).item()
        # Eager modda derleme/JIT YOKTUR; bu yüzden etiket de "derleme dahil" demez.
        run_label = (
            "PyTorch compiled first training run (torch.compile dahil)"
            if compiled
            else "PyTorch eager first training run"
        )
        print(f"  {run_label:<56}: {first_run_time:5.2f} s  (test_acc={display_acc*100:.2f}%)")

        # (b) Sıfırla + UNTIMED "gösterim" koşusu (ölçüme dahil DEĞİL).
        demo_model, demo_optimizer = reset_fn()
        run_torch_epochs(
            demo_model, demo_optimizer, step_fn, x_train_dev, y_train_labels_dev, x_test_dev, y_test_labels_dev, idx_dev,
            label=f"PyTorch {label}", verbose=True,
        )

        if name != "GPU":
            steady_times, _ = time_it(
                lambda: run_torch_and_sync(
                    *reset_fn(), step_fn, x_train_dev, y_train_labels_dev, x_test_dev, y_test_labels_dev, idx_dev,
                )
            )
            results["CPU"] = (first_run_time, steady_times, display_acc)
            print_stats(f"PyTorch ({label}) {name} steady-state", steady_times)
            continue

        # =====================================================================
        # GPU: iki ayrı, birbirine KARIŞTIRILMAMASI gereken ölçüm.
        # =====================================================================

        # --- Compute-only: reset_fn() ZATEN GPU'da (host transferi YOK). ---
        gpu_compute_times, _ = time_it(
            lambda: run_torch_and_sync(
                *reset_fn(), step_fn, x_train_dev, y_train_labels_dev, x_test_dev, y_test_labels_dev, idx_dev,
            )
        )
        results["GPU compute-only"] = (first_run_time, gpu_compute_times, display_acc)
        print_stats(f"PyTorch ({label}) GPU compute-only", gpu_compute_times)

        # --- End-to-end (round-trip): her tekrarda HOST verisinden/ağırlık-
        # larından başlanır -> GPU'ya taşınır -> 5 epoch GPU'da eğitilir ->
        # son doğruluk host'a geri getirilir. ---
        # NOT: burada `prepare_torch` YENİDEN ÇAĞRILMAZ. Yeni bir model/optimizer/
        # derlenmiş sarmalayıcı oluşturmak `torch.compile` guard'larını (özellikle
        # `check_obj_id(optimizer, ...)`) düşürür, her tekrarda yeniden derlemeye ve
        # `recompile_limit` aşımına yol açardı. Round-trip semantiği korunur: veri
        # host'tan cihaza kopyalanır ve pristine ağırlıklar HOST'tan yüklenir.
        def round_trip():
            x_train_iter = torch.from_numpy(x_train).to(device)
            y_train_labels_iter = torch.from_numpy(y_train_labels.astype(np.int64)).to(device)
            x_test_iter = torch.from_numpy(x_test).to(device)
            y_test_labels_iter = torch.from_numpy(y_test_labels.astype(np.int64)).to(device)
            idx_iter = torch.from_numpy(idx_array.astype(np.int64)).to(device)

            model_iter, optimizer_iter = load_from_host_fn()  # HOST -> GPU ağırlık transferi BURADA

            final_model_iter, _ = run_torch_epochs(
                model_iter, optimizer_iter, step_fn, x_train_iter, y_train_labels_iter, x_test_iter, y_test_labels_iter, idx_iter,
            )
            acc_iter = accuracy_torch(final_model_iter, x_test_iter, y_test_labels_iter)
            torch_sync(device)
            return acc_iter.cpu().item()  # GPU -> host senkron + kopya

        round_trip()  # warmup: allocator/transfer yollarını ısıt (derleme zaten hazır)
        gpu_e2e_times, e2e_acc = time_it(round_trip)
        results["GPU end-to-end"] = (gpu_e2e_times, e2e_acc)
        print_stats(f"PyTorch ({label}) GPU end-to-end", gpu_e2e_times)

    return results


# ===========================================================================
# 4) FLAX NNX + OPTAX + JAX
#
#    NE ELLE YAZILMIYOR (kaldırılan boilerplate) VE KİM SAĞLIYOR:
#      - Model/layer/parametre-durumu organizasyonu -> Flax NNX
#        (parametreler artık elle dict DEĞİL, `nnx.Module` İÇİNDE yaşıyor)
#      - Geri yayılım (backward pass)                -> JAX/NNX autodiff
#        (`nnx.value_and_grad`; zincir kuralı elle YAZILMIYOR)
#      - Optimizer / update mantığı                  -> Optax
#        (`optax.sgd` + `nnx.Optimizer`; elle `p - lr*g` YAZILMIYOR)
#      - Training step derlemesi                      -> NNX/JAX JIT
#        (`@nnx.jit`)
# ===========================================================================
class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(784, HIDDEN_DIM, rngs=rngs)
        self.fc2 = nnx.Linear(HIDDEN_DIM, 10, rngs=rngs)

    def __call__(self, x):
        x = jnp.tanh(self.fc1(x))
        return self.fc2(x)


@nnx.jit
def train_step_nnx(model: MLP, optimizer: nnx.Optimizer, x, y_labels):
    def loss_fn(model):
        logits = model(x)
        # Optax'ın integer-label softmax cross-entropy'si: y_onehot'a gerek yok.
        return optax.softmax_cross_entropy_with_integer_labels(logits, y_labels).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)  # backward pass OTOMATİK (NumPy'daki dz2/dW2/... YOK)
    optimizer.update(model, grads)  # SGD update OTOMATİK (NumPy'daki "p - lr*g" YOK)
    return loss


@nnx.jit
def accuracy_nnx(model: MLP, x, y_labels):
    logits = model(x)
    preds = jnp.argmax(logits, axis=1)
    return jnp.mean(preds == y_labels)


def run_nnx_epochs(model, optimizer, x_train_dev, y_train_labels_dev, x_test_dev, y_test_labels_dev, idx_dev, name="", verbose=False):
    """SADECE hesaplama: tüm girdiler ZATEN cihazda kabul edilir. `verbose=False`
    iken döngü içinde HİÇBİR host<->device transferi/senkronizasyonu YAPILMAZ
    (epoch kaybı device üzerinde jnp skaleri olarak biriktirilir).

    Flax >= 0.11: `nnx.Optimizer` artık `model` referansı TUTMAZ; bu yüzden
    `model` ve `optimizer` ayrı argümanlar olarak geçirilir/döndürülür."""
    epochs, n_batches, _ = idx_dev.shape
    for epoch in range(epochs):
        epoch_loss = jnp.array(0.0)
        for b in range(n_batches):
            batch_idx = idx_dev[epoch, b]
            loss = train_step_nnx(model, optimizer, x_train_dev[batch_idx], y_train_labels_dev[batch_idx])
            epoch_loss = epoch_loss + loss  # device üzerinde kalır, host'a senkron OLMAZ
        if verbose:
            test_acc = float(accuracy_nnx(model, x_test_dev, y_test_labels_dev))
            print(f"  [Flax NNX] Epoch {epoch+1}/{EPOCHS}  loss={float(epoch_loss)/n_batches:.4f}  test_acc={test_acc*100:.2f}%")
    return model, optimizer


def run_and_sync(model, optimizer, *args, **kwargs):
    """`run_nnx_epochs`'u çalıştırır ve GERÇEKTEN cihaz hesaplamasının
    bittiğinden emin olur. `jax.block_until_ready` ham nnx.Module/Optimizer
    nesnelerini DEĞİL, `nnx.state(...)`'in ürettiği düz JAX Pytree'sini
    güvenle dolaşabilir; bu yüzden senkronizasyon `nnx.state` üzerinden yapılır."""
    model, optimizer = run_nnx_epochs(model, optimizer, *args, **kwargs)
    jax.block_until_ready(nnx.state(model))
    jax.block_until_ready(nnx.state(optimizer))
    return model, optimizer


def prepare_nnx(device, init_host, seed=0):
    """Modeli/optimizer'ı HOST init ağırlıklarından cihaza BİR KEZ taşır ve İKİ
    AYRI "pristine" (el değmemiş) durum anlık görüntüsü çıkarır — Flax >= 0.11'de
    `nnx.Optimizer` artık `model` referansı TUTMAZ, bu yüzden `nnx.state(optimizer)`
    yalnızca opt_state'i içerir, model ağırlıklarını İÇERMEZ. Döndürülen `reset_fn`,
    her çağrıldığında HEM modeli HEM optimizer'ı kendi pristine durumuna GERİ
    YÜKLER — bu işlem tamamen cihaz-içidir, HOST transferi GEREKTİRMEZ (compute-only
    ölçümü kirletmez)."""
    with jax.default_device(device):
        model = MLP(rngs=nnx.Rngs(seed))
        # NNX'in kendi (rastgele) ilklendirmesini, NumPy tarafıyla BİREBİR AYNI
        # olan `init_host` ağırlıklarıyla EZİYORUZ (adil/eşit başlangıç):
        # Güncel Flax NNX API'sinde `Variable.value` getter/setter'ı DEPRECATED;
        # array Variable'lar için `variable[...]` ile okunur, `variable[...] = x`
        # ile yazılır.
        model.fc1.kernel[...] = jnp.asarray(init_host["W1"])
        model.fc1.bias[...] = jnp.asarray(init_host["b1"])
        model.fc2.kernel[...] = jnp.asarray(init_host["W2"])
        model.fc2.bias[...] = jnp.asarray(init_host["b2"])
        optimizer = nnx.Optimizer(model, optax.sgd(LEARNING_RATE), wrt=nnx.Param)

    # nnx.state(...) TEK BAŞINA yeterli DEĞİL: geri dönen State, altta yatan
    # Variable'larla referans PAYLAŞABİLİR — bu durumda `optimizer.update()`
    # canlı modeli değiştirdiğinde "pristine" anlık görüntü de SESSİZCE
    # kirlenir (reset hiçbir şey yapmıyormuş gibi görünür). `nnx.clone` ile
    # GERÇEKTEN bağımsız bir kopya alınır.
    pristine_model_state = nnx.clone(nnx.state(model))          # yalnızca ağırlıklar, cihazda, BAĞIMSIZ kopya
    pristine_optimizer_state = nnx.clone(nnx.state(optimizer))   # yalnızca opt_state, cihazda, BAĞIMSIZ kopya

    def reset_fn():
        nnx.update(model, pristine_model_state)          # cihaz-içi reset, host transferi YOK
        nnx.update(optimizer, pristine_optimizer_state)   # cihaz-içi reset, host transferi YOK
        return model, optimizer

    return reset_fn


def benchmark_flax_implementation(devices, init_host, x_train, y_train_labels, x_test, y_test_labels, idx_array):
    """Flax NNX + Optax için ölçüm iskeleti — `benchmark_torch_implementation`
    ile AYNI metodoloji (sanity check -> first run -> untimed demo -> 10x
    steady-state; GPU'da compute-only + end-to-end AYRI). Dönüş:
    {'CPU'|'GPU compute-only': (first_run, times, acc), 'GPU end-to-end': (times, acc)}."""
    results = {}

    for name, device in devices.items():
        print("\n" + "-" * 62)
        print(f"  Flax NNX + Optax — {name}")
        print("-" * 62)

        x_train_dev = jax.device_put(x_train, device)
        y_train_labels_dev = jax.device_put(y_train_labels, device)
        x_test_dev = jax.device_put(x_test, device)
        y_test_labels_dev = jax.device_put(y_test_labels, device)
        idx_dev = jax.device_put(idx_array, device)

        reset_fn = prepare_nnx(device, init_host)

        # --- Sanity check (TEK sefer, ölçüm dışı) ---
        sanity_model, _ = reset_fn()
        assert np.allclose(np.asarray(sanity_model.fc1.kernel[...]), init_host["W1"]), f"[Flax/{name}] pristine W1 NumPy init ile eşleşmiyor!"
        assert np.allclose(np.asarray(sanity_model.fc1.bias[...]), init_host["b1"]), f"[Flax/{name}] pristine b1 NumPy init ile eşleşmiyor!"
        assert np.allclose(np.asarray(sanity_model.fc2.kernel[...]), init_host["W2"]), f"[Flax/{name}] pristine W2 NumPy init ile eşleşmiyor!"
        assert np.allclose(np.asarray(sanity_model.fc2.bias[...]), init_host["b2"]), f"[Flax/{name}] pristine b2 NumPy init ile eşleşmiyor!"
        print(f"  [{name}] Sanity check: reset sonrası ağırlıklar NumPy init ile eşleşiyor. ✓")

        # (a) "first training run": PRISTINE durumdan, verbose=False, HENÜZ
        # hiçbir train_step_nnx/accuracy_nnx çağrısı yapılmamışken ölçülür
        # -> GERÇEKTEN JIT derlemesini + tam 5 epoch'u birlikte içerir.
        first_run_times, (first_model, first_optimizer) = time_it(
            lambda: run_and_sync(
                *reset_fn(), x_train_dev, y_train_labels_dev, x_test_dev, y_test_labels_dev, idx_dev,
            ),
            repeat=1,
        )
        first_run_time = first_run_times[0]
        display_acc = float(accuracy_nnx(first_model, x_test_dev, y_test_labels_dev))
        print(f"  {'Flax/JAX first training run (includes JIT compilation)':<56}: {first_run_time:5.2f} s  (test_acc={display_acc*100:.2f}%)")

        # (b) Sıfırla + UNTIMED "gösterim" koşusu (JIT zaten sıcak; ölçüme dahil DEĞİL).
        demo_model, demo_optimizer = reset_fn()
        run_nnx_epochs(
            demo_model, demo_optimizer, x_train_dev, y_train_labels_dev, x_test_dev, y_test_labels_dev, idx_dev,
            name=name, verbose=True,
        )

        if name != "GPU":
            cpu_times, _ = time_it(
                lambda: run_and_sync(
                    *reset_fn(), x_train_dev, y_train_labels_dev, x_test_dev, y_test_labels_dev, idx_dev,
                )
            )
            results["CPU"] = (first_run_time, cpu_times, display_acc)
            print_stats(f"Flax/JAX {name} steady-state", cpu_times)
            continue

        # =====================================================================
        # GPU: iki ayrı, birbirine KARIŞTIRILMAMASI gereken ölçüm.
        # =====================================================================

        # --- Compute-only: reset_fn() ZATEN GPU'da (host transferi YOK). ---
        gpu_compute_times, _ = time_it(
            lambda: run_and_sync(
                *reset_fn(), x_train_dev, y_train_labels_dev, x_test_dev, y_test_labels_dev, idx_dev,
            )
        )
        results["GPU compute-only"] = (first_run_time, gpu_compute_times, display_acc)
        print_stats("Flax/JAX GPU compute-only", gpu_compute_times)

        # --- End-to-end (round-trip): her tekrarda HOST verisinden/ağırlık-
        # larından başlanır -> GPU'ya taşınır -> 5 epoch GPU'da eğitilir ->
        # son doğruluk `jax.device_get` ile host'a geri getirilir. ---
        def round_trip():
            reset_fn_iter = prepare_nnx(device, init_host)  # host -> GPU transferi BURADA
            x_train_iter = jax.device_put(x_train, device)
            y_train_labels_iter = jax.device_put(y_train_labels, device)
            x_test_iter = jax.device_put(x_test, device)
            y_test_labels_iter = jax.device_put(y_test_labels, device)
            idx_iter = jax.device_put(idx_array, device)

            final_model_iter, final_optimizer_iter = run_nnx_epochs(
                *reset_fn_iter(), x_train_iter, y_train_labels_iter, x_test_iter, y_test_labels_iter, idx_iter,
            )
            acc_iter = accuracy_nnx(final_model_iter, x_test_iter, y_test_labels_iter)
            return jax.device_get(acc_iter)  # GPU -> host senkron + kopya

        round_trip()  # warmup: allocator/transfer yollarını ısıt (JIT zaten hazır)
        gpu_e2e_times, e2e_acc = time_it(round_trip)
        results["GPU end-to-end"] = (gpu_e2e_times, float(e2e_acc))
        print_stats("Flax/JAX GPU end-to-end", gpu_e2e_times)

    return results


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
    mean_s = statistics.mean(times)
    std_s = statistics.stdev(times)
    print(f"  {label:<28}: {mean_s:5.2f} ± {std_s:4.2f} s (n={len(times)})")


# ---------------------------------------------------------------------------
# Özet yazdırma: PyTorch (eager/compiled) ve Flax sonuç sözlükleri AYNI
# şekle sahiptir ({'CPU'|'GPU compute-only': (first_run, times, acc),
# 'GPU end-to-end': (times, acc)}); bu yüzden TEK bir fonksiyonla yazdırılır.
# ---------------------------------------------------------------------------
def print_summary_block(title, entries, numpy_mean):
    print(f"\n{title}")
    if not entries:
        print("  (bu ortamda/platformda çalıştırılamadı, atlandı)")
        return
    for label, values in entries.items():
        if len(values) == 3:
            first_run, times, acc = values
        else:
            times, acc = values
            first_run = None
        mean_s = statistics.mean(times)
        std_s = statistics.stdev(times)
        print(f"  [{label}] final accuracy : {acc*100:.2f}%")
        if first_run is not None:
            print(f"  [{label}] first training run : {first_run:.2f} s")
        print(f"  [{label}] steady-state        : {mean_s:.2f} ± {std_s:.2f} s (n={len(times)})")
        print(f"  [{label}] speedup vs NumPy    : {numpy_mean / mean_s:.1f}x")


if __name__ == "__main__":
    print("=" * 62)
    print(" DENEY 6: NumPy vs PyTorch (eager/compiled) vs Flax NNX + Optax")
    print("=" * 62)

    devices = get_available_devices()
    torch_devices = get_available_torch_devices()
    print(f"  Bulunan JAX cihazları    : {list(devices.keys())}")
    print(f"  Bulunan PyTorch cihazları: {list(torch_devices.keys())}")
    print_tf32_note()

    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_mnist()

    rng = np.random.default_rng(0)
    x_train, y_train_labels, y_train_onehot = preprocess(x_train_full, y_train_full, N_TRAIN, rng)
    x_test, y_test_labels, _ = preprocess(x_test_full, y_test_full, N_TEST, rng)

    print(f"\n  Eğitim örneği : {N_TRAIN:,}".replace(",", "."))
    print(f"  Test örneği   : {N_TEST:,}".replace(",", "."))
    print(f"  Mimari        : 784 -> {HIDDEN_DIM} (tanh) -> 10 (softmax)")
    print(f"  Epoch / Batch : {EPOCHS} epoch, batch boyutu {BATCH_SIZE}")

    init = init_params(np.random.default_rng(42))  # DÖRT yöntem de AYNI başlangıçtan başlar
    idx_array = build_epoch_indices(N_TRAIN, BATCH_SIZE, EPOCHS, seed=1)  # DÖRT yöntem AYNI batch sırasını kullanır

    # =======================================================================
    # 1) NumPy ile eğitim (elle backprop)
    # =======================================================================
    print("\n" + "-" * 62)
    print("  [1] NumPy (elle türetilmiş backpropagation)")
    print("-" * 62)

    numpy_final_acc = run_numpy_training(init, x_train, y_train_onehot, x_test, y_test_labels, idx_array, verbose=True)
    numpy_times, _ = time_it(
        lambda: run_numpy_training(init, x_train, y_train_onehot, x_test, y_test_labels, idx_array, verbose=False)
    )
    numpy_mean = statistics.mean(numpy_times)

    # =======================================================================
    # 2) PyTorch (eager)
    # =======================================================================
    print("\n" + "=" * 62)
    print("  [2] PyTorch (eager) — nn.Module + autograd + torch.optim.SGD")
    print("=" * 62)
    pytorch_eager_results = benchmark_torch_implementation(
        "eager", False, torch_devices, init, x_train, y_train_labels, x_test, y_test_labels, idx_array,
    )

    # =======================================================================
    # 3) PyTorch (compiled) — platform/sürüm destekliyorsa
    # =======================================================================
    print("\n" + "=" * 62)
    print("  [3] PyTorch (compiled) — torch.compile")
    print("=" * 62)
    if hasattr(torch, "compile"):
        pytorch_compiled_results = benchmark_torch_implementation(
            "compiled", True, torch_devices, init, x_train, y_train_labels, x_test, y_test_labels, idx_array,
        )
    else:
        print("  torch.compile bu PyTorch sürümünde mevcut değil (PyTorch >= 2.0 gerekir);")
        print("  'PyTorch (compiled)' atlanıyor.")
        pytorch_compiled_results = {}

    # =======================================================================
    # 4) Flax NNX + Optax
    # =======================================================================
    print("\n" + "=" * 62)
    print("  [4] Flax NNX + Optax + JAX")
    print("=" * 62)
    flax_results = benchmark_flax_implementation(
        devices, init, x_train, y_train_labels, x_test, y_test_labels, idx_array,
    )

    # =======================================================================
    # Özet
    # =======================================================================
    print("\n" + "=" * 62)
    print(" ÖZET (ortalama ± std, n=10)")
    print("=" * 62)

    print("\nNumPy manual backprop")
    print(f"  final accuracy: {numpy_final_acc*100:.2f}%")
    print(f"  steady-state  : {statistics.mean(numpy_times):.2f} ± {statistics.stdev(numpy_times):.2f} s (n={len(numpy_times)})")

    print_summary_block("PyTorch (eager)", pytorch_eager_results, numpy_mean)
    print_summary_block("PyTorch (compiled)", pytorch_compiled_results, numpy_mean)
    print_summary_block("Flax NNX + Optax", flax_results, numpy_mean)

    slow_labels = []
    for impl_name, entries in [("PyTorch (eager)", pytorch_eager_results), ("PyTorch (compiled)", pytorch_compiled_results), ("Flax NNX + Optax", flax_results)]:
        if "CPU" in entries:
            cpu_mean = statistics.mean(entries["CPU"][1])
            if cpu_mean > numpy_mean:
                slow_labels.append(f"{impl_name} [CPU]")

    if slow_labels:
        print(
            f"\n  Not: {', '.join(slow_labels)} burada NumPy'dan YAVAŞ çıkabilir — bu bir\n"
            "  hata değil, önemli bir ders! Batch boyutu (100) çok küçük olduğundan,\n"
            "  her adımdaki asıl hesaplama ucuz kalıyor; buna karşılık her batch için\n"
            "  ayrı bir derlenmiş/otograd adımı dispatch etme maliyeti (Python\n"
            "  tarafında), kazanılan hesaplama zamanını aşabiliyor. Derleme (JIT /\n"
            "  torch.compile) büyük batch'lerde, GPU/TPU'da veya tüm epoch döngüsü\n"
            "  tek bir grafikte (ör. `jax.lax.scan`) derlenip Python dispatch'i\n"
            "  ortadan kaldırıldığında asıl avantajını gösterir. Bu deneyin amacı\n"
            "  zaten hız değil, boilerplate azaltımıdır (bkz. dosya başındaki not)."
        )

    print("=" * 62)
