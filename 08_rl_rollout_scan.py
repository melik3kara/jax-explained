"""
Deney 8: RL Rollout Sistemleri — jit / vmap / lax.scan Kompozisyonu
=====================================================================
Bu deney "JAX hızlı mı?" sorusunu DEĞİL, şu hipotezi kontrollü biçimde test eder:

    JAX'in jit + vmap + lax.scan kombinasyonu, küçük fixed-state ve yüksek
    paralellik içeren RL environment'ları için çok uygun olabilir; ancak daha
    sequential veya memory-intensive environment'larda aynı avantaj küçülebilir.

Üç JAX transformation'ı doğrudan bir RL sisteminin üç eksenine eşlenir:

    jit   -> tek bir numerical computation'ın (bir env adımının) derlenmesi
    vmap  -> environment ekseni paralelliği (N bağımsız env aynı anda)
    scan  -> zaman ekseni recurrence'ı (T adımlık rollout tek bir loop'a füzyon)

ve bunların birlikte kullanımı = "compiled batched rollout".

## Test edilen iki synthetic environment (kasıtlı olarak ZIT tasarlandı)

  A) SmallStateParallelEnv   — JAX'in doğal olarak uygun olduğu workload
     - 8 float32'lik sabit, küçük state (x, v, theta, omega, target, energy,
       phase, t)
     - saf numerical transition, düşük branch sayısı, I/O yok
     - env'ler birbirinden tamamen bağımsız -> batch'lenmesi kolay

  B) LargeStateSequentialEnv — "large-state, scattered-access, sequentially
     dependent workload" (bilerek "memory-bound" DEMİYORUZ: bunu iddia etmek
     profiler ölçümü gerektirir, bu deneyde yapılmadı)
     - state'in büyük kısmı büyük bir float32 memory buffer'ı
     - her adımda buffer'dan birkaç SCATTERED index okunur, SADECE BİR eleman
       yazılır (compute-light, erişim örüntüsü dağınık)
     - pointer/accumulator önceki adıma güçlü şekilde bağımlıdır -> zaman ekseni
       gerçekten sequential'dır, paralelleştirilemez
     - batch büyüdükçe çalışma kümesi (working set) hızla büyür

Her iki env de FIXED SHAPE kullanır (JAX derleyebilsin diye); dinamik Python
listesi yoktur.

## Karşılaştırılan dört rollout implementation'ı (matematiksel olarak AYNI)

  A_np  NumPy baseline      : env ekseni vektörize, zaman ekseni Python `for`
  B_jit jit(vmap(step)) + Python zaman döngüsü : her timestep AYRI bir dispatch
  C_scan jit(scan(step))   : zaman ekseni füzyonlu ama env ekseni Python
                              döngüsünde (N ayrı rollout çağrısı)
  D_all jit(scan(vmap(step))) : her iki eksen de tek bir derlenmiş çağrıda

Bu dörtlü kasıtlı olarak bir ablation'dır:
    B vs D  -> zaman ekseni füzyonunun (scan) katkısı
    C vs D  -> env ekseni füzyonunun (vmap) katkısı
    A vs D  -> toplam kazanç, dürüst bir NumPy referansına karşı

NumPy baseline'ı hakkında ÖNEMLİ not: baseline BİLEREK "her env için ayrı bir
Python skaler döngüsü" şeklinde YAZILMADI. Öyle bir baseline saman adam (straw
man) olur ve JAX'i haksız yere iyi gösterirdi. Bunun yerine gerçek bir NumPy
kullanıcısının yazacağı hâli kullanıyoruz: env ekseni vektörize, zaman ekseni
Python döngüsünde ve memory buffer YERİNDE (in-place) güncelleniyor — ki bu son
nokta, functional update yapmak zorunda olan JAX'e karşı NumPy'ın LEHİNEdir.

## Ölçüm metodolojisi (repo'nun 04/05 numaralı deneyleriyle AYNI)
  - first call (tracing + derleme + çalıştırma) `number=1` ile AYRI ölçülür ve
    steady-state'e karıştırılmaz.
  - steady-state: otomatik iç döngü kalibrasyonu (~200 ms'lik ölçüm penceresi),
    `timeit.repeat(repeat=10)`, çağrı başına normalize, mean ± sample std.
  - her timed JAX çağrısı `jax.block_until_ready(...)` ile senkronize edilir.
  - CPU ve (varsa) GPU ayrı ayrı; GPU'da compute-only ve end-to-end AYRI.
  - önce correctness, sonra performans: dört implementation da aynı initial
    state / aynı action dizisiyle çalıştırılıp final state, reward trajectory ve
    done flag'leri karşılaştırılmadan HİÇBİR performans sonucu basılmaz.
  - hızlanma yoksa "slower" olarak açıkça raporlanır; sonuç JAX lehine
    zorlanmaz.

## Literatür bağlamı
  - PureJaxRL (Lu et al.), environment'ın KENDİSİNİ de JAX'e taşıyıp tüm
    training pipeline'ını (env adımı dahil) tek bir derlenmiş, vmap'lenmiş
    hesaba dönüştürme fikrini popülerleştirdi; buradaki D implementation'ı bu
    desenin küçük bir modelidir.
  - Karten, Appapogu & Jin, "Automatic Generation of High-Performance RL
    Environments", arXiv:2603.12145 — optimization guide'ında şunu söylüyor:
    "When the training loop calls env.step inside a rollout loop, fuse the loop
    with jax.lax.scan to compile the entire rollout into one kernel." ve
    CartPole için: "In CartPole, this improved throughput by 3.2x over a Python
    loop calling jitted steps."
    Aynı çalışma backend seçimi için de şunu belirtiyor: "We select between JAX
    and Rust based on environment structure: JAX suits environments with small
    state memory that benefit from parallel execution on GPU; Rust suits
    sequential or memory-intensive environments."

    Buradaki 3.2x SADECE karşılaştırılacak bir literatür değeridir; bu deneyde
    o sayıyı elde etmeyi BEKLEMİYORUZ ve o yönde zorlamıyoruz. Ayrıca bu deney
    Karten et al.'ın JAX-vs-Rust iddiasının TAMAMINI kanıtlamaz; yalnızca iki
    kontrollü synthetic workload ile trade-off'un JAX tarafını inceler.
    (Opsiyonel Rust CPU baseline'ı, sistemde Rust toolchain'i varsa raporlanır;
    yoksa deney bundan bağımsız olarak tamamlanır.)
"""

import math
import shutil
import statistics
import timeit

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# Benchmark ayarları
# ---------------------------------------------------------------------------
REPEATS = 10
TARGET_SECONDS = 0.2          # her timeit tekrarının hedeflenen ölçüm penceresi
MIN_INNER_LOOPS = 1
MAX_INNER_LOOPS = 100_000

NUM_ENVS_LIST = (1, 32, 256, 1024)
ROLLOUT_LENGTHS = (128, 1024)

# Tek bir steady-state çağrısı bu süreyi aşarsa o konfigürasyon ATLANIR (10
# tekrar × N saniye kabul edilemez uzunlukta olurdu). Atlananlar çıktıda
# AÇIKÇA raporlanır — sessizce gizlenmez.
MAX_SECONDS_PER_CALL = 2.5

# Correctness kontrolü için küçük konfigürasyon
CHECK_ENVS = 8
CHECK_STEPS = 64
CHECK_RTOL = 1e-4
CHECK_ATOL = 1e-4


# ---------------------------------------------------------------------------
# Ortak sabitler — NumPy ve JAX tarafında AYNI np.float32 skalerleri kullanılır
# (Python float'ı kullanmak NumPy tarafında sessizce float64'e yükseltip iki
# implementation arasında yapay bir sayısal fark yaratırdı).
# ---------------------------------------------------------------------------
F32 = np.float32

# --- Environment A sabitleri ---
STATE_DIM_A = 8
DT_A = F32(0.02)
SPRING_K = F32(1.5)
DAMPING = F32(0.1)
DRIVE = F32(0.5)
TORQUE_GAIN = F32(0.3)
GRAVITY = F32(0.2)
ACTION_COST = F32(0.01)
X_LIMIT = F32(6.0)
EPISODE_LIMIT_A = F32(500.0)

# --- Environment B sabitleri ---
# Memory buffer uzunluğu (fixed shape). İlk çalıştırma 4096 ile yapıldı ve
# ÖLÇÜLDÜ Kİ o boyutta beklenen memory-bandwidth baskısı OLUŞMUYOR: N=1024'te
# çalışma kümesi 16 MiB'de kalıyor ve her adımda 4096 elemandan yalnızca 4'ü
# okunup/yazıldığı için Env B, Env A'dan bile hızlı çıkıyordu. 16384'e
# çıkarmak N=1024'te çalışma kümesini ~64 MiB'ye taşır (tipik L3'ün üstü) ve
# hipotezin "memory-intensive" ucunu gerçekten test eder. Bu bir sonuç
# ayarlaması DEĞİL, deney koşulunun kurulmasıdır: 4096'lık sonuç da aşağıdaki
# "SINIRLILIKLAR" notunda açıkça anlatılmıştır.
STATE_SIZE_B = 16384          # memory buffer uzunluğu (fixed shape)
STRIDE_B = 3907               # asal-benzeri stride -> scattered okuma
MIX_A = F32(0.5)
MIX_B = F32(0.3)
MIX_C = F32(0.2)
ACC_DECAY = F32(0.99)
ACC_THRESHOLD = F32(0.35)
DAMPEN = F32(0.5)
REWARD_ACC_COST = F32(0.01)
EPISODE_LIMIT_B = 500


# ===========================================================================
# Benchmark yardımcıları — Deney 04/05 ile AYNI (repo genelinde tutarlı)
# ===========================================================================
def choose_inner_loops(
    single_call_seconds: float,
    target_seconds: float = TARGET_SECONDS,
    min_number: int = MIN_INNER_LOOPS,
    max_number: int = MAX_INNER_LOOPS,
) -> int:
    """Her timeit tekrarının ~`target_seconds` sürmesi için gereken iç döngü sayısı.

    YAPILAN İŞİ DEĞİŞTİRMEZ: aynı rollout `number` kez tekrarlanır ve toplam süre
    `number`'a bölünerek çağrı başına süreye normalize edilir. Bu deneydeki
    rollout'lar genelde ms-s mertebesinde olduğundan `number` çoğunlukla 1 çıkar;
    hızlı konfigürasyonlarda (küçük N, kısa T) ise timer gürültüsünü bastırır."""
    if single_call_seconds <= 0:
        return max_number
    number = round(target_seconds / single_call_seconds)
    return int(max(min_number, min(number, max_number)))


def benchmark_steady(
    fn,
    repeats: int = REPEATS,
    target_seconds: float = TARGET_SECONDS,
    single_call_seconds: float = None,
) -> dict:
    """Kalibre edilmiş iç döngüyle steady-state ölçüm. fn'in ÖNCEDEN sıcak olması
    beklenir (JAX için ayrıca ölçülen first call). Dönen süreler ÇAĞRI BAŞINA'dır.

    `single_call_seconds` verilirse kalibrasyon çağrısı TEKRARLANMAZ (çağıran
    zaten bir probe ölçümü yapmıştır)."""
    result = None

    def call():
        nonlocal result
        result = fn()

    if single_call_seconds is None:
        single_call_seconds = timeit.timeit(call, number=1)

    number = choose_inner_loops(single_call_seconds, target_seconds)
    totals = timeit.repeat(call, number=number, repeat=repeats)
    per_call_times = [total / number for total in totals]

    return {
        "times": per_call_times,
        "number": number,
        "mean": statistics.mean(per_call_times),
        "std": statistics.stdev(per_call_times) if len(per_call_times) > 1 else 0.0,
        "result": result,
    }


def fmt_ms(seconds: float) -> str:
    ms = seconds * 1000
    return f"{ms:.3f}" if ms < 1 else f"{ms:.2f}"


def fmt_sps(sps: float) -> str:
    """Steps-per-second'ı okunabilir biçimde (K/M/G) yazdır."""
    if sps >= 1e9:
        return f"{sps/1e9:.2f}G"
    if sps >= 1e6:
        return f"{sps/1e6:.2f}M"
    if sps >= 1e3:
        return f"{sps/1e3:.1f}K"
    return f"{sps:.0f}"


def relative_performance(baseline: float, candidate: float) -> str:
    """Yavaşlamayı '0.1x hızlı' gibi anlamsız biçimde DEĞİL, açıkça raporla."""
    if candidate <= baseline:
        return f"{baseline / candidate:.1f}x faster"
    return f"{candidate / baseline:.1f}x slower"


def break_even_repeats(first_call_time: float, steady_mean: float, baseline_mean: float):
    """first_call + (n-1)*steady <= n*baseline  =>  n >= (first_call-steady)/(baseline-steady)"""
    if baseline_mean <= steady_mean:
        return None  # steady-state baseline'dan hızlı değil; asla amorti olmaz
    n = (first_call_time - steady_mean) / (baseline_mean - steady_mean)
    return max(1, math.ceil(n))


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


# ===========================================================================
# ENVIRONMENT A — SmallStateParallelEnv
#
# State (8 x float32): [x, v, theta, omega, target, energy, phase, t]
# Action: tek bir float32 (sürekli kuvvet)
#
# Sentetik dinamik (harici bir RL kütüphanesine bağımlılık YOK): hedefe doğru
# yaylı-sönümlü bir kütle + sürülen bir sarkaç açısı. Saf numerical, dallanma
# neredeyse yok, env'ler birbirinden tamamen bağımsız.
#
# NOT (kasıtlı tasarım): `done` olduğunda otomatik reset YAPILMAZ. Reset,
# dört implementation'a da ek bir koşullu yol eklerdi; bu deneyin ölçmek
# istediği şey rollout füzyonu, episode yönetimi değil. `done` yalnızca
# raporlanır ve correctness'ta karşılaştırılır.
# ===========================================================================
def init_states_a(n_envs: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = np.zeros((n_envs, STATE_DIM_A), dtype=np.float32)
    state[:, 0] = rng.uniform(-1.0, 1.0, n_envs)      # x
    state[:, 1] = rng.uniform(-0.5, 0.5, n_envs)      # v
    state[:, 2] = rng.uniform(-0.3, 0.3, n_envs)      # theta
    state[:, 3] = rng.uniform(-0.2, 0.2, n_envs)      # omega
    state[:, 4] = rng.uniform(-2.0, 2.0, n_envs)      # target
    state[:, 5] = 0.0                                  # energy
    state[:, 6] = rng.uniform(0.0, 6.28, n_envs)      # phase
    state[:, 7] = 0.0                                  # t
    return state


def make_actions_a(n_envs: int, n_steps: int, seed: int = 1) -> np.ndarray:
    """(T, N) float32 action dizisi — TÜM implementation'lar AYNI diziyi kullanır."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n_steps, n_envs)).astype(np.float32)


def step_a_np(state: np.ndarray, action: np.ndarray):
    """NumPy, env ekseni VEKTÖRİZE. state: (N, 8), action: (N,)."""
    x, v, th, om = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    target, phase, tstep = state[:, 4], state[:, 6], state[:, 7]

    force = action - DAMPING * v + DRIVE * np.sin(phase)
    v_new = v + DT_A * (force - SPRING_K * (x - target))
    x_new = x + DT_A * v_new
    om_new = om + DT_A * (TORQUE_GAIN * action - GRAVITY * np.sin(th))
    th_new = th + DT_A * om_new
    energy_new = F32(0.5) * v_new * v_new + F32(0.5) * SPRING_K * (x_new - target) ** 2
    phase_new = phase + DT_A
    t_new = tstep + F32(1.0)

    next_state = np.empty_like(state)
    next_state[:, 0] = x_new
    next_state[:, 1] = v_new
    next_state[:, 2] = th_new
    next_state[:, 3] = om_new
    next_state[:, 4] = target
    next_state[:, 5] = energy_new
    next_state[:, 6] = phase_new
    next_state[:, 7] = t_new

    reward = (-((x_new - target) ** 2) - ACTION_COST * action * action).astype(np.float32)
    done = (np.abs(x_new) > X_LIMIT) | (t_new >= EPISODE_LIMIT_A)
    return next_state, reward, done


def step_a_jax(state, action):
    """JAX, TEK env. state: (8,), action: skaler. Saf fonksiyon.

    `vmap` env ekseni için, `scan` zaman ekseni için bu fonksiyonun ÜZERİNE
    uygulanır — fonksiyonun kendisi hiçbir batch/zaman bilgisi taşımaz."""
    x, v, th, om, target, _energy, phase, tstep = (
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]
    )

    force = action - DAMPING * v + DRIVE * jnp.sin(phase)
    v_new = v + DT_A * (force - SPRING_K * (x - target))
    x_new = x + DT_A * v_new
    om_new = om + DT_A * (TORQUE_GAIN * action - GRAVITY * jnp.sin(th))
    th_new = th + DT_A * om_new
    energy_new = F32(0.5) * v_new * v_new + F32(0.5) * SPRING_K * (x_new - target) ** 2
    phase_new = phase + DT_A
    t_new = tstep + F32(1.0)

    next_state = jnp.stack([x_new, v_new, th_new, om_new, target, energy_new, phase_new, t_new])
    reward = -((x_new - target) ** 2) - ACTION_COST * action * action
    done = (jnp.abs(x_new) > X_LIMIT) | (t_new >= EPISODE_LIMIT_A)
    return next_state, reward, done


# ===========================================================================
# ENVIRONMENT B — LargeStateSequentialEnv
#
# State: (memory[4096] float32, pointer int32, accumulator float32, t int32)
# Action: (float32 değer, int32 pointer-ilerleme miktarı ∈ {0,1})
#
# Her adım:
#   - memory'den 3 SCATTERED index okunur (biri pointer'ın kendisi)
#   - action ile karıştırılıp tanh'tan geçirilir
#   - accumulator'a bağlı bir KOŞULLU sönümleme uygulanır (jnp.where / np.where)
#   - SADECE pointer konumuna yazılır  (compute-light, scattered access)
#   - accumulator güncellenir, pointer VERİYE BAĞLI olarak ilerler
#
# Bu yapı zaman eksenini gerçekten sequential yapar: t adımındaki okuma,
# t-1'de yazılmış olabilecek hücreye bağlıdır.
#
# NOT (correctness'ı sağlam tutmak için bilinçli tercih): pointer ilerlemesi
# HESAPLANAN float değere değil, önceden üretilmiş `advance` action'ına bağlıdır.
# Float bir eşiğe (`val > 0` gibi) bağlansaydı, NumPy ile JAX arasındaki ~1e-7
# mertebesindeki normal float32 farkı bir adımda dalı ters çevirip iki
# implementation'ı tamamen ayırabilirdi — ölçmek istediğimiz şey bu değil.
# Değere bağlı koşul, `acc > ACC_THRESHOLD` sönümlemesinde KORUNUR.
# ===========================================================================
def init_states_b(n_envs: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    memory = rng.uniform(-0.5, 0.5, (n_envs, STATE_SIZE_B)).astype(np.float32)
    pointer = rng.integers(0, STATE_SIZE_B, n_envs).astype(np.int32)
    acc = np.zeros(n_envs, dtype=np.float32)
    t = np.zeros(n_envs, dtype=np.int32)
    return memory, pointer, acc, t


def make_actions_b(n_envs: int, n_steps: int, seed: int = 1):
    """(T, N) float32 değer + (T, N) int32 pointer-ilerleme."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(-1.0, 1.0, (n_steps, n_envs)).astype(np.float32)
    advance = rng.integers(0, 2, (n_steps, n_envs)).astype(np.int32)
    return values, advance


def step_b_np(state, action, advance):
    """NumPy, env ekseni VEKTÖRİZE, memory YERİNDE (in-place) güncellenir.

    In-place güncelleme NumPy'ın LEHİNEdir: JAX functional update (`.at[].set()`)
    kullanmak zorundadır. Bu asimetri bilerek korunuyor — baseline'ı yapay
    olarak zayıflatmıyoruz."""
    memory, pointer, acc, t = state
    n = memory.shape[0]
    rows = np.arange(n)

    i0 = pointer
    i1 = (pointer * 7 + 13) % STATE_SIZE_B
    i2 = (pointer + STRIDE_B) % STATE_SIZE_B

    a = memory[rows, i0]
    b = memory[rows, i1]
    c = memory[rows, i2]

    mixed = MIX_A * a + MIX_B * b + MIX_C * c
    val = np.tanh(mixed + action).astype(np.float32)
    val = np.where(acc > ACC_THRESHOLD, val * DAMPEN, val).astype(np.float32)

    memory[rows, i0] = val                      # yalnızca 1 hücre yazılır
    acc_new = (acc * ACC_DECAY + val).astype(np.float32)
    pointer_new = ((pointer + 1 + advance) % STATE_SIZE_B).astype(np.int32)
    t_new = (t + 1).astype(np.int32)

    reward = (val - REWARD_ACC_COST * np.abs(acc_new)).astype(np.float32)
    done = t_new >= EPISODE_LIMIT_B
    return (memory, pointer_new, acc_new, t_new), reward, done


def step_b_jax(state, action, advance):
    """JAX, TEK env. memory: (STATE_SIZE_B,), pointer/acc/t: skaler."""
    memory, pointer, acc, t = state

    i0 = pointer
    i1 = (pointer * 7 + 13) % STATE_SIZE_B
    i2 = (pointer + STRIDE_B) % STATE_SIZE_B

    a = memory[i0]
    b = memory[i1]
    c = memory[i2]

    mixed = MIX_A * a + MIX_B * b + MIX_C * c
    val = jnp.tanh(mixed + action)
    val = jnp.where(acc > ACC_THRESHOLD, val * DAMPEN, val)

    memory_new = memory.at[i0].set(val)         # functional update
    acc_new = acc * ACC_DECAY + val
    pointer_new = (pointer + 1 + advance) % STATE_SIZE_B
    t_new = t + 1

    reward = val - REWARD_ACC_COST * jnp.abs(acc_new)
    done = t_new >= EPISODE_LIMIT_B
    return (memory_new, pointer_new, acc_new, t_new), reward, done


# ===========================================================================
# ENV SPEC — iki environment'ı tek bir benchmark iskeletinden sürebilmek için
# ===========================================================================
class EnvSpec:
    def __init__(self, key, name, init_fn, actions_fn, step_np, step_jax, describe):
        self.key = key
        self.name = name
        self.init_fn = init_fn
        self.actions_fn = actions_fn
        self.step_np = step_np
        self.step_jax = step_jax
        self.describe = describe


def _describe_a():
    return f"state: {STATE_DIM_A} x float32 (küçük, sabit) | action: 1 x float32"


def _describe_b():
    kib = STATE_SIZE_B * 4 / 1024
    return (f"large-state, scattered-access, sequentially dependent workload | "
            f"state: memory[{STATE_SIZE_B}] float32 ({kib:.0f} KiB/env) + pointer + acc + t "
            f"| action: 1 x float32 + 1 x int32")


ENV_A = EnvSpec("A", "SmallStateParallelEnv", init_states_a, make_actions_a,
                step_a_np, step_a_jax, _describe_a)
ENV_B = EnvSpec("B", "LargeStateSequentialEnv", init_states_b, make_actions_b,
                step_b_np, step_b_jax, _describe_b)


# ===========================================================================
# DÖRT ROLLOUT IMPLEMENTATION'I
#
# Hepsi AYNI çıktıyı üretir: (final_state, rewards (T,N), dones (T,N))
# ===========================================================================

# --- A) NumPy baseline: env ekseni vektörize, zaman ekseni Python for --------
def rollout_numpy(spec, init_state, actions):
    if spec.key == "A":
        state = init_state.copy()
        n_steps = actions.shape[0]
        rewards = np.empty(actions.shape, dtype=np.float32)
        dones = np.empty(actions.shape, dtype=bool)
        for t in range(n_steps):
            state, r, d = step_a_np(state, actions[t])
            rewards[t] = r
            dones[t] = d
        return state, rewards, dones

    memory, pointer, acc, tt = init_state
    values, advance = actions
    n_steps = values.shape[0]
    state = (memory.copy(), pointer.copy(), acc.copy(), tt.copy())
    rewards = np.empty(values.shape, dtype=np.float32)
    dones = np.empty(values.shape, dtype=bool)
    for t in range(n_steps):
        state, r, d = step_b_np(state, values[t], advance[t])
        rewards[t] = r
        dones[t] = d
    return state, rewards, dones


# --- B) jit(vmap(step)) + Python zaman döngüsü ------------------------------
def make_rollout_jit_step_pyloop(spec):
    """Her TIMESTEP için ayrı bir derlenmiş çağrı dispatch edilir.

    Senkronizasyon: adım BAŞINA `block_until_ready` YAPILMAZ — gerçek bir
    kullanıcı da yapmaz ve yapmak yapay bir maliyet eklerdi. Bunun yerine
    rollout'un SONUNDA bir kez senkronize edilir (liste de bir pytree'dir,
    `block_until_ready` tüm yaprakları bekler); wall-clock ölçümü böylece hem
    doğru hem de bu yöntemin LEHİNEdir.

    ÖNEMLİ (ölçüm sınırı, bilinçli olarak B'nin LEHİNE): trajectory burada
    (T, N) tek bir dizi hâline GETİRİLMEZ; per-step çıktılar Python listesinde
    kalır ve stack'leme ölçüm DIŞINDA (`assemble_pyloop`) yapılır. Sebep:
    T=1024 ayrı diziyi `jnp.stack` ile birleştirmek 1024 operand'lı dev bir XLA
    concatenate üretir ve tek başına DAKİKALARCA derlenir — bu, ölçmek
    istediğimiz rollout maliyeti değil, çıktı toplama biçiminin patolojisidir.
    D yöntemi ise trajectory'yi scan'in `ys` çıktısı olarak ölçüm İÇİNDE
    üretir; yani bu tercih D'yi değil B'yi kayırır."""
    if spec.key == "A":
        step_batched = jax.jit(jax.vmap(step_a_jax, in_axes=(0, 0)))

        def rollout(init_state, actions):
            state = init_state
            rewards, dones = [], []
            for t in range(actions.shape[0]):
                state, r, d = step_batched(state, actions[t])
                rewards.append(r)
                dones.append(d)
            return jax.block_until_ready((state, rewards, dones))
    else:
        step_batched = jax.jit(jax.vmap(step_b_jax, in_axes=(0, 0, 0)))

        def rollout(init_state, actions):
            values, advance = actions
            state = init_state
            rewards, dones = [], []
            for t in range(values.shape[0]):
                state, r, d = step_batched(state, values[t], advance[t])
                rewards.append(r)
                dones.append(d)
            return jax.block_until_ready((state, rewards, dones))

    return rollout


def assemble_pyloop(spec, raw):
    """B yönteminin ham çıktısını (T, N) NumPy dizilerine çevirir — ÖLÇÜM DIŞI,
    yalnızca correctness karşılaştırması için."""
    state, rewards, dones = raw
    final = np.asarray(state) if spec.key == "A" else tuple(np.asarray(x) for x in state)
    return final, np.stack([np.asarray(r) for r in rewards]), np.stack([np.asarray(d) for d in dones])


# --- C) jit(scan(step)) — zaman füzyonlu, env ekseni Python döngüsünde -------
def make_single_env_scan(spec):
    """TEK bir environment için tüm T adımlık rollout'u lax.scan ile füzyonlar."""
    if spec.key == "A":
        def body(state, action):
            next_state, r, d = step_a_jax(state, action)
            return next_state, (r, d)

        @jax.jit
        def rollout_one(init_state, actions):
            final_state, (rewards, dones) = lax.scan(body, init_state, actions)
            return final_state, rewards, dones
    else:
        def body(state, inputs):
            action, advance = inputs
            next_state, r, d = step_b_jax(state, action, advance)
            return next_state, (r, d)

        @jax.jit
        def rollout_one(init_state, actions):
            final_state, (rewards, dones) = lax.scan(body, init_state, actions)
            return final_state, rewards, dones

    return rollout_one


def make_rollout_scan_per_env(spec, n_envs):
    """N environment -> N AYRI derlenmiş scan çağrısı (env ekseni Python'da)."""
    rollout_one = make_single_env_scan(spec)

    # Ölçüm sınırı B ile AYNI: env ekseninde stack'leme ölçüm DIŞINDA
    # (`assemble_per_env`) yapılır — bkz. `make_rollout_jit_step_pyloop`.
    if spec.key == "A":
        def rollout(init_state, actions):
            finals, rewards, dones = [], [], []
            for e in range(n_envs):
                fs, r, d = rollout_one(init_state[e], actions[:, e])
                finals.append(fs)
                rewards.append(r)
                dones.append(d)
            return jax.block_until_ready((finals, rewards, dones))
    else:
        def rollout(init_state, actions):
            memory, pointer, acc, tt = init_state
            values, advance = actions
            finals, rewards, dones = [], [], []
            for e in range(n_envs):
                fs, r, d = rollout_one(
                    (memory[e], pointer[e], acc[e], tt[e]),
                    (values[:, e], advance[:, e]),
                )
                finals.append(fs)
                rewards.append(r)
                dones.append(d)
            return jax.block_until_ready((finals, rewards, dones))

    return rollout


def assemble_per_env(spec, raw):
    """C yönteminin ham çıktısını (N, ...) / (T, N) NumPy dizilerine çevirir —
    ÖLÇÜM DIŞI, yalnızca correctness karşılaştırması için."""
    finals, rewards, dones = raw
    if spec.key == "A":
        final = np.stack([np.asarray(f) for f in finals])
    else:
        final = tuple(np.stack([np.asarray(f[i]) for f in finals]) for i in range(4))
    return (final,
            np.stack([np.asarray(r) for r in rewards], axis=1),
            np.stack([np.asarray(d) for d in dones], axis=1))


# --- D) jit(scan(vmap(step))) — her iki eksen tek derlenmiş çağrıda ----------
def make_rollout_vmap_scan_jit(spec):
    """Zaman ekseni scan, env ekseni vmap, tamamı tek bir jit'li fonksiyon.

    `vmap(scan(...))` kompozisyonu da matematiksel olarak EŞDEĞERdir (env'ler
    bağımsız); burada PureJaxRL'de yaygın olan `scan(vmap(step))` sırası
    kullanılıyor: zaman döngüsünün gövdesi batch'lenmiş bir env adımıdır."""
    if spec.key == "A":
        step_batched = jax.vmap(step_a_jax, in_axes=(0, 0))

        def body(state, action):
            next_state, r, d = step_batched(state, action)
            return next_state, (r, d)

        @jax.jit
        def rollout(init_state, actions):
            final_state, (rewards, dones) = lax.scan(body, init_state, actions)
            return final_state, rewards, dones
    else:
        step_batched = jax.vmap(step_b_jax, in_axes=(0, 0, 0))

        def body(state, inputs):
            action, advance = inputs
            next_state, r, d = step_batched(state, action, advance)
            return next_state, (r, d)

        @jax.jit
        def rollout(init_state, actions):
            values, advance = actions
            final_state, (rewards, dones) = lax.scan(body, init_state, (values, advance))
            return final_state, rewards, dones

    def wrapped(init_state, actions):
        return jax.block_until_ready(rollout(init_state, actions))

    return wrapped


# ===========================================================================
# Cihaza taşıma / host'a getirme yardımcıları
# ===========================================================================
def put_env(spec, init_state, actions, device):
    if spec.key == "A":
        return jax.device_put(init_state, device), jax.device_put(actions, device)
    memory, pointer, acc, t = init_state
    values, advance = actions
    dev_state = (
        jax.device_put(memory, device), jax.device_put(pointer, device),
        jax.device_put(acc, device), jax.device_put(t, device),
    )
    return dev_state, (jax.device_put(values, device), jax.device_put(advance, device))


def to_numpy_result(spec, result):
    """(final_state, rewards, dones) -> saf NumPy, correctness karşılaştırması için."""
    final_state, rewards, dones = result
    if spec.key == "A":
        final_np = np.asarray(final_state)
    else:
        final_np = tuple(np.asarray(leaf) for leaf in final_state)
    return final_np, np.asarray(rewards), np.asarray(dones)


# ===========================================================================
# CORRECTNESS — performans sonuçlarından ÖNCE
# ===========================================================================
def max_abs_diff(ref, cand):
    if isinstance(ref, tuple):
        return max(max_abs_diff(r, c) for r, c in zip(ref, cand))
    return float(np.max(np.abs(np.asarray(ref, dtype=np.float64) - np.asarray(cand, dtype=np.float64))))


def allclose_tree(ref, cand, rtol=CHECK_RTOL, atol=CHECK_ATOL) -> bool:
    """Pytree'nin HER yaprağını ayrı ayrı karşılaştırır. Yaprakları tek bir
    vektörde birleştirmek yanıltıcı olurdu: Env B'nin final state'inde hem
    ~1e-1 mertebesinde float memory değerleri hem de ~4096'ya kadar çıkan int
    pointer'lar var; ortak bir rtol eşiği float tarafını gereksiz gevşetirdi."""
    if isinstance(ref, tuple):
        return all(allclose_tree(r, c, rtol, atol) for r, c in zip(ref, cand))
    return bool(np.allclose(np.asarray(ref, dtype=np.float64),
                            np.asarray(cand, dtype=np.float64), rtol=rtol, atol=atol))


def check_correctness(spec, devices) -> bool:
    print(f"\n  Correctness (N={CHECK_ENVS}, T={CHECK_STEPS}, referans = NumPy baseline):")

    init_state = spec.init_fn(CHECK_ENVS, seed=7)
    actions = spec.actions_fn(CHECK_ENVS, CHECK_STEPS, seed=11)

    ref = rollout_numpy(spec, init_state, actions)
    ref_final, ref_rewards, ref_dones = ref

    all_ok = True
    for dev_name, device in devices.items():
        dev_state, dev_actions = put_env(spec, init_state, actions, device)
        candidates = {
            "B jit-step+pyloop": (make_rollout_jit_step_pyloop(spec),
                                  lambda raw: assemble_pyloop(spec, raw)),
            "C scan (per-env)": (make_rollout_scan_per_env(spec, CHECK_ENVS),
                                 lambda raw: assemble_per_env(spec, raw)),
            "D vmap+scan+jit": (make_rollout_vmap_scan_jit(spec),
                                lambda raw: to_numpy_result(spec, raw)),
        }
        for label, (fn, assemble) in candidates.items():
            final, rewards, dones = assemble(fn(dev_state, dev_actions))
            d_state = max_abs_diff(ref_final, final)
            d_rew = max_abs_diff(ref_rewards, rewards)
            done_ok = bool(np.array_equal(ref_dones, dones))
            ok = (
                np.allclose(ref_rewards, rewards, rtol=CHECK_RTOL, atol=CHECK_ATOL)
                and allclose_tree(ref_final, final)
                and done_ok
            )
            all_ok = all_ok and ok
            mark = "✓" if ok else "✗ UYUŞMUYOR"
            print(f"    [{dev_name}] {label:<20}: final Δ={d_state:.2e}  reward Δ={d_rew:.2e}  "
                  f"done eşit={done_ok}  {mark}")

    return all_ok


# ===========================================================================
# Tek bir (yöntem, N, T, cihaz) konfigürasyonunu ölçen sürücü
# ===========================================================================
def measure(fn, args, n_envs, n_steps, single_env_hint=None):
    """first call (compile+run, number=1) -> probe -> steady-state.

    `single_env_hint`: C yöntemi için, N env'lik maliyet N ile DOĞRUSAL
    büyüdüğünden, N=1'de ölçülen maliyetten tahmin edilir; tahmin bütçeyi
    aşarsa konfigürasyon HİÇ çalıştırılmadan atlanır (aksi hâlde tek bir probe
    dakikalarca sürebilirdi)."""
    total_steps = n_envs * n_steps

    if single_env_hint is not None:
        predicted = single_env_hint * n_envs
        if predicted > MAX_SECONDS_PER_CALL:
            return {"skipped": f"tahmini ~{predicted:.1f} s/çağrı > {MAX_SECONDS_PER_CALL:.1f} s bütçe"}

    first_call = timeit.timeit(lambda: fn(*args), number=1)
    probe = timeit.timeit(lambda: fn(*args), number=1)

    if probe > MAX_SECONDS_PER_CALL:
        return {"skipped": f"steady çağrısı ~{probe:.1f} s > {MAX_SECONDS_PER_CALL:.1f} s bütçe",
                "first_call": first_call}

    stats = benchmark_steady(lambda: fn(*args), single_call_seconds=probe)
    stats["first_call"] = first_call
    stats["sps"] = total_steps / stats["mean"]
    return stats


def row(env_key, n_envs, n_steps, method, device, stats):
    if "skipped" in stats:
        first = fmt_ms(stats["first_call"]) if "first_call" in stats else "-"
        print(f"  {n_envs:>5} | {n_steps:>5} | {method:<22} | {device:<4} | {first:>9} | "
              f"{'ATLANDI':>18} | {'-':>9}   ({stats['skipped']})")
        return
    print(f"  {n_envs:>5} | {n_steps:>5} | {method:<22} | {device:<4} | "
          f"{fmt_ms(stats['first_call']):>9} | "
          f"{fmt_ms(stats['mean']) + ' ± ' + fmt_ms(stats['std']):>18} | "
          f"{fmt_sps(stats['sps']):>9}")


HEADER = (f"  {'N':>5} | {'T':>5} | {'method':<22} | {'dev':<4} | {'first(ms)':>9} | "
          f"{'steady(ms ± std)':>18} | {'SPS':>9}")


# ===========================================================================
# Bir environment için tüm matrisi çalıştır
# ===========================================================================
def run_env_benchmarks(spec, devices):
    print("\n" + "=" * 100)
    print(f"  ## Environment {spec.key}: {spec.name}")
    print(f"  {spec.describe()}")
    print("=" * 100)
    print(HEADER)
    print("  " + "-" * 96)

    results = {}   # (n_envs, n_steps, method, device) -> stats
    scan_single_cost = {}  # (n_steps, device) -> N=1 scan maliyeti (C tahmini için)

    for n_steps in ROLLOUT_LENGTHS:
        for n_envs in NUM_ENVS_LIST:
            init_state = spec.init_fn(n_envs, seed=3)
            actions = spec.actions_fn(n_envs, n_steps, seed=5)

            # --- A) NumPy baseline (cihazdan bağımsız) ---
            rollout_numpy(spec, init_state, actions)   # untimed warmup
            np_stats = benchmark_steady(lambda: rollout_numpy(spec, init_state, actions))
            np_stats["first_call"] = float("nan")
            np_stats["sps"] = n_envs * n_steps / np_stats["mean"]
            results[(n_envs, n_steps, "A NumPy baseline", "CPU")] = np_stats
            print(f"  {n_envs:>5} | {n_steps:>5} | {'A NumPy baseline':<22} | {'CPU':<4} | "
                  f"{'-':>9} | {fmt_ms(np_stats['mean']) + ' ± ' + fmt_ms(np_stats['std']):>18} | "
                  f"{fmt_sps(np_stats['sps']):>9}")

            # --- JAX yöntemleri, her cihazda ---
            for dev_name, device in devices.items():
                dev_state, dev_actions = put_env(spec, init_state, actions, device)

                b_fn = make_rollout_jit_step_pyloop(spec)
                b_stats = measure(b_fn, (dev_state, dev_actions), n_envs, n_steps)
                results[(n_envs, n_steps, "B jit-step+pyloop", dev_name)] = b_stats
                row(spec.key, n_envs, n_steps, "B jit-step+pyloop", dev_name, b_stats)

                hint = scan_single_cost.get((n_steps, dev_name)) if n_envs > 1 else None
                c_fn = make_rollout_scan_per_env(spec, n_envs)
                c_stats = measure(c_fn, (dev_state, dev_actions), n_envs, n_steps,
                                  single_env_hint=hint)
                results[(n_envs, n_steps, "C scan (per-env)", dev_name)] = c_stats
                row(spec.key, n_envs, n_steps, "C scan (per-env)", dev_name, c_stats)
                if n_envs == 1 and "skipped" not in c_stats:
                    scan_single_cost[(n_steps, dev_name)] = c_stats["mean"]

                d_fn = make_rollout_vmap_scan_jit(spec)
                d_stats = measure(d_fn, (dev_state, dev_actions), n_envs, n_steps)
                results[(n_envs, n_steps, "D vmap+scan+jit", dev_name)] = d_stats
                row(spec.key, n_envs, n_steps, "D vmap+scan+jit", dev_name, d_stats)

                # --- GPU end-to-end: host -> GPU -> rollout -> host ---
                # Sadece D için ölçülüyor; action dizisi (T,N) burada gerçekten
                # büyük olduğundan transfer maliyeti anlamlıdır.
                if dev_name == "GPU":
                    d_e2e = make_rollout_vmap_scan_jit(spec)

                    def round_trip():
                        s_dev, a_dev = put_env(spec, init_state, actions, device)
                        out = d_e2e(s_dev, a_dev)
                        return jax.device_get(out)

                    # `d_e2e` YENİ bir jit'li fonksiyondur; untimed warmup YAPILMAZ ki
                    # ölçülen first call diğer yöntemlerle tutarlı biçimde
                    # (tracing + derleme + transfer + çalıştırma) olsun.
                    e2e_stats = measure(round_trip, (), n_envs, n_steps)
                    results[(n_envs, n_steps, "D end-to-end", "GPU")] = e2e_stats
                    row(spec.key, n_envs, n_steps, "D end-to-end", "GPU", e2e_stats)

            print("  " + "-" * 96)

    return results


# ===========================================================================
# Otomatik, dürüst yorum üretimi
# ===========================================================================
def summarize(spec, results, devices):
    print(f"\n  --- Environment {spec.key} ({spec.name}) özeti ---")

    # (1) scan ablation: B (Python loop + jitted step) vs D (scan fused)
    print("\n  [1] scan ablation — B (Python loop + jitted step) vs D (scan-fused rollout):")
    ratios = []
    for dev_name in devices:
        for n_steps in ROLLOUT_LENGTHS:
            for n_envs in NUM_ENVS_LIST:
                b = results.get((n_envs, n_steps, "B jit-step+pyloop", dev_name))
                d = results.get((n_envs, n_steps, "D vmap+scan+jit", dev_name))
                if not b or not d or "skipped" in b or "skipped" in d:
                    continue
                ratio = b["mean"] / d["mean"]
                ratios.append((ratio, dev_name, n_envs, n_steps))
                verdict = (f"scan was {ratio:.1f}x faster than Python-controlled jitted stepping"
                           if ratio >= 1.0 else
                           f"scan was {1/ratio:.1f}x slower than Python-controlled jitted stepping")
                print(f"      [{dev_name}] N={n_envs:>4} T={n_steps:>4}: {verdict}")
    if ratios:
        best = max(ratios)
        worst = min(ratios)
        print(f"      -> aralık: {worst[0]:.1f}x ({worst[1]}, N={worst[2]}, T={worst[3]}) "
              f"... {best[0]:.1f}x ({best[1]}, N={best[2]}, T={best[3]})")
        print("      -> karşılaştırma için: Karten et al. (arXiv:2603.12145) CartPole'da 3.2x "
              "rapor ediyor;")
        print("         bu deneyin sayısı o değere eşit OLMAK ZORUNDA DEĞİLDİR (farklı env, "
              "farklı donanım).")

    # (2) vmap ölçeklenmesi
    print("\n  [2] vmap ölçeklenmesi (D yöntemi, SPS'in N ile değişimi):")
    for dev_name in devices:
        for n_steps in ROLLOUT_LENGTHS:
            line = []
            base = None
            for n_envs in NUM_ENVS_LIST:
                d = results.get((n_envs, n_steps, "D vmap+scan+jit", dev_name))
                if not d or "skipped" in d:
                    line.append(f"N={n_envs}: -")
                    continue
                if base is None:
                    base = d["sps"]
                line.append(f"N={n_envs}: {fmt_sps(d['sps'])} ({d['sps']/base:.1f}x)")
            print(f"      [{dev_name}] T={n_steps:>4}: " + "  ".join(line))

    # (3) CPU vs GPU
    if "GPU" in devices and "CPU" in devices:
        print("\n  [3] CPU vs GPU (D yöntemi, compute-only steady-state):")
        for n_steps in ROLLOUT_LENGTHS:
            for n_envs in NUM_ENVS_LIST:
                c = results.get((n_envs, n_steps, "D vmap+scan+jit", "CPU"))
                g = results.get((n_envs, n_steps, "D vmap+scan+jit", "GPU"))
                if not c or not g or "skipped" in c or "skipped" in g:
                    continue
                print(f"      N={n_envs:>4} T={n_steps:>4}: GPU is "
                      f"{relative_performance(c['mean'], g['mean'])} than CPU")
        print("\n      GPU end-to-end (host->GPU->rollout->host) vs GPU compute-only:")
        for n_steps in ROLLOUT_LENGTHS:
            for n_envs in NUM_ENVS_LIST:
                co = results.get((n_envs, n_steps, "D vmap+scan+jit", "GPU"))
                e2 = results.get((n_envs, n_steps, "D end-to-end", "GPU"))
                if not co or not e2 or "skipped" in co or "skipped" in e2:
                    continue
                overhead = (e2["mean"] - co["mean"]) / co["mean"] * 100
                print(f"      N={n_envs:>4} T={n_steps:>4}: transfer ek yükü "
                      f"%{overhead:.0f}  ({fmt_ms(co['mean'])} -> {fmt_ms(e2['mean'])} ms)")

    # (4) NumPy'a karşı ve break-even
    print("\n  [4] D (vmap+scan+jit) vs NumPy baseline + derleme break-even'ı:")
    for dev_name in devices:
        for n_steps in ROLLOUT_LENGTHS:
            for n_envs in NUM_ENVS_LIST:
                npy = results.get((n_envs, n_steps, "A NumPy baseline", "CPU"))
                d = results.get((n_envs, n_steps, "D vmap+scan+jit", dev_name))
                if not npy or not d or "skipped" in d:
                    continue
                be = break_even_repeats(d["first_call"], d["mean"], npy["mean"])
                be_str = f"~{be} rollout" if be is not None else "asla (steady-state NumPy'dan yavaş)"
                print(f"      [{dev_name}] N={n_envs:>4} T={n_steps:>4}: "
                      f"{relative_performance(npy['mean'], d['mean']):<16} | "
                      f"first call {fmt_ms(d['first_call'])} ms | break-even: {be_str}")


def compare_environments(results_a, results_b, devices):
    """Env A ve Env B'nin BATCHING'den ne kadar faydalandığını karşılaştır."""
    print("\n" + "=" * 100)
    print("  ## Environment A vs Environment B — batching kazancı")
    print("=" * 100)
    n_lo, n_hi = NUM_ENVS_LIST[0], NUM_ENVS_LIST[-1]
    print(f"  Ölçüt: 'batching verimliliği' = SPS(N={n_hi}) / SPS(N={n_lo}) oranının, MÜKEMMEL")
    print(f"  lineer ölçeklenmeye (={n_hi}x) göre yüzdesi. %100 = env eklemek env başına")
    print("  maliyeti hiç artırmıyor; %25 = paralellikten faydanın dörtte biri alınabiliyor.")
    print()
    print("  NEDEN ham 'kaç kat arttı' oranı DEĞİL: o oran N=1'deki başlangıç noktasını")
    print("  ödüllendirir. N=1'de zaten kötü olan bir environment daha çok 'kat' kazanır")
    print("  ama bu onu batching için daha uygun YAPMAZ. Bu yüzden hem ideal-lineerliğe")
    print("  göre verimlilik hem de MUTLAK tepe SPS birlikte raporlanır.\n")

    eff = {}
    peak = {}
    for label, results in (("A (SmallStateParallel)", results_a), ("B (LargeStateSequential)", results_b)):
        for dev_name in devices:
            for n_steps in ROLLOUT_LENGTHS:
                lo = results.get((n_lo, n_steps, "D vmap+scan+jit", dev_name))
                hi = results.get((n_hi, n_steps, "D vmap+scan+jit", dev_name))
                if not lo or not hi or "skipped" in lo or "skipped" in hi:
                    continue
                factor = hi["sps"] / lo["sps"]
                efficiency = factor / n_hi * 100.0
                eff[(label, dev_name, n_steps)] = efficiency
                peak[(label, dev_name, n_steps)] = hi["sps"]
                print(f"    {label:<26} [{dev_name}] T={n_steps:>4}: "
                      f"{fmt_sps(lo['sps'])} -> {fmt_sps(hi['sps'])} SPS  "
                      f"({factor:.1f}x = ideal lineerliğin %{efficiency:.0f}'i)")

    print()
    for dev_name in devices:
        for n_steps in ROLLOUT_LENGTHS:
            key_a = ("A (SmallStateParallel)", dev_name, n_steps)
            key_b = ("B (LargeStateSequential)", dev_name, n_steps)
            ea, eb = eff.get(key_a), eff.get(key_b)
            if ea is None or eb is None:
                continue
            pa, pb = peak[key_a], peak[key_b]
            if ea > eb * 1.05:
                verdict = (f"Environment A benefited more from batching than Environment B "
                           f"(%{ea:.0f} vs %{eb:.0f} of ideal)")
            elif eb > ea * 1.05:
                verdict = (f"Environment B benefited more from batching than Environment A "
                           f"(%{eb:.0f} vs %{ea:.0f} of ideal)")
            else:
                verdict = (f"her iki environment de batching'den PRATİKTE AYNI ölçüde faydalandı "
                           f"(%{ea:.0f} vs %{eb:.0f} of ideal)")
            print(f"    [{dev_name}] T={n_steps:>4}: {verdict};")
            print(f"                  tepe SPS: A={fmt_sps(pa)} vs B={fmt_sps(pb)} "
                  f"({relative_performance(1.0/pa, 1.0/pb)} — B, A'ya göre)")

    if "CPU" in devices:
        print()
        print("    UYARI (CPU satırlarını okurken): CPU'da ideal-lineerlik yüzdesi zaten")
        print("    çekirdek sayısıyla sınırlıdır; 1024 env'i lineer ölçeklendirmek FİZİKSEL")
        print("    olarak mümkün değildir. Bu yüzden CPU'daki yüzde, 'batching'e uygunluk'tan")
        print("    çok 'N=1'de ne kadar boşta kapasite kalmıştı'yı ölçer ve düşük N=1")
        print("    performansını ödüllendirir. Hipotez açısından ASIL BAKILACAK satırlar")
        print("    GPU satırları ile her iki cihazdaki MUTLAK tepe SPS değerleridir.")

    print()
    print("  SINIRLILIKLAR (bu deneyin ölçemedikleri):")
    print("   - Env B 'memory-intensive'i buffer BOYUTU ile modeller, ama her adımda")
    print(f"     {STATE_SIZE_B} elemandan yalnızca 4'ünü okuyup 1'ini yazar. Gerçek bir")
    print("     bandwidth-bound workload adım başına buffer'ın BÜYÜK bir kısmına dokunurdu;")
    print("     o rejim burada ÖLÇÜLMEMİŞTİR. (İlk denemede buffer 4096 idi; o boyutta")
    print("     hiçbir bandwidth baskısı gözlenmedi ve Env B, Env A'dan hızlı çıktı.)")
    print("   - Sequential bağımlılık her iki env'de de ZAMAN ekseninde; scan zaten zaman")
    print("     eksenini paralelleştirmeye ÇALIŞMAZ, sadece füzyonlar. Yani bu deney")
    print("     'sequential olmak scan'i yavaşlatır mı?' sorusunu değil, 'sequential bir")
    print("     env'de scan füzyonu ve vmap hâlâ işe yarıyor mu?' sorusunu ölçer.")
    print("   - Tek bir makine, tek GPU (laptop sınıfı), tek JAX sürümü. Farklı bellek")
    print("     hiyerarşisi olan bir sistemde A/B farkı değişebilir.")
    print("   - Rust tarafı ölçülmediği için 'JAX mı Rust mı' sorusu BURADAN cevaplanamaz.")
    print("\n  Bu iki sentetik workload, Karten et al. (arXiv:2603.12145) çalışmasının backend")
    print("  seçimi hakkındaki iddiasının TAMAMINI kanıtlamaz veya çürütmez; yalnızca")
    print("  trade-off'un JAX tarafını iki kontrollü uçta ölçer. Yukarıdaki sayılar bu")
    print("  makinede, bu env tanımlarıyla geçerlidir; genel bir 'JAX RL için her zaman")
    print("  daha iyidir/kötüdür' sonucu ÇIKARILAMAZ.")


# ===========================================================================
# Opsiyonel Rust CPU baseline'ı — varsa raporla, yoksa deneyi bozma
# ===========================================================================
def numpy_vs_jax_tables(results_a, results_b):
    """NumPy CPU baseline vs JAX jit(scan(vmap(...))) CPU — sunum tablosu.

    YENİ ÖLÇÜM YAPMAZ: `run_env_benchmarks` içinde ZATEN alınmış olan
    ("A NumPy baseline", "CPU") ve ("D vmap+scan+jit", "CPU") kayıtlarını okur.
    Böylece warmup/kalibrasyon/repeat=10/correctness mantığı aynen korunur ve
    aynı iş ikinci kez ölçülmez.

    Bu tablo, scan ablation'dan (B jit-step+pyloop vs D scan) AYRI bir sorudur
    ve onunla karıştırılmamalıdır: burada karşılaştırılan şey JAX'in iki farklı
    yazım biçimi değil, NumPy ile JAX'tir."""
    print("\n" + "=" * 100)
    print("  ## NumPy CPU baseline  vs  JAX jit(scan(vmap(step))) CPU")
    print("=" * 100)
    print("  Baseline adaleti: NumPy tarafında ENV EKSENİ tam vektörizedir (env başına")
    print("  ayrı Python döngüsü YOKTUR); yalnızca ZAMAN EKSENİ Python-controlled bir")
    print("  döngüdür — JAX tarafındaki scan'in füzyonladığı eksen de budur. Ayrıca")
    print("  NumPy memory buffer'ı IN-PLACE günceller, JAX ise functional update")
    print("  (`.at[].set()`) kullanmak zorundadır; bu asimetri NumPy'ın LEHİNEdir.")
    print("  speedup = numpy_ms / jax_ms   (>1 => JAX hızlı, <1 => JAX yavaş)")

    summary = {}
    for label, results in (("A (SmallStateParallelEnv)", results_a),
                           ("B (LargeStateSequentialEnv)", results_b)):
        for n_steps in ROLLOUT_LENGTHS:
            print(f"\n  Environment {label} — CPU, T={n_steps}")
            print("  " + "-" * 96)
            print(f"  {'N':>6} | {'NumPy ms':>11} | {'JAX scan+vmap ms':>17} | "
                  f"{'Speedup vs NumPy':>18} | {'NumPy SPS':>11} | {'JAX SPS':>11}")
            print("  " + "-" * 96)
            for n_envs in NUM_ENVS_LIST:
                npy = results.get((n_envs, n_steps, "A NumPy baseline", "CPU"))
                jx = results.get((n_envs, n_steps, "D vmap+scan+jit", "CPU"))
                if not npy or not jx or "skipped" in npy or "skipped" in jx:
                    print(f"  {n_envs:>6} | {'-':>11} | {'-':>17} | {'ölçülemedi':>18} | "
                          f"{'-':>11} | {'-':>11}")
                    continue
                speedup = npy["mean"] / jx["mean"]
                verdict = (f"{speedup:.2f}x faster" if speedup >= 1.0
                           else f"{speedup:.2f}x / {1.0/speedup:.2f}x slower")
                summary[(label, n_steps, n_envs)] = speedup
                print(f"  {n_envs:>6} | {npy['mean']*1000:>11.3f} | {jx['mean']*1000:>17.3f} | "
                      f"{verdict:>18} | {fmt_sps(npy['sps']):>11} | {fmt_sps(jx['sps']):>11}")
            print("  " + "-" * 96)

    # --- A vs B yan yana ---
    print("\n" + "=" * 100)
    print("  ## Environment A vs Environment B — NumPy'a karşı JAX avantajı nasıl değişiyor?")
    print("=" * 100)
    print(f"  {'T':>6} | {'N':>6} | {'A: JAX vs NumPy':>18} | {'B: JAX vs NumPy':>18} | {'A avantajı / B avantajı':>26}")
    print("  " + "-" * 96)
    for n_steps in ROLLOUT_LENGTHS:
        for n_envs in NUM_ENVS_LIST:
            sa = summary.get(("A (SmallStateParallelEnv)", n_steps, n_envs))
            sb = summary.get(("B (LargeStateSequentialEnv)", n_steps, n_envs))
            if sa is None or sb is None:
                continue
            ratio = sa / sb
            note = (f"A {ratio:.2f}x daha avantajlı" if ratio >= 1.0
                    else f"B {1.0/ratio:.2f}x daha avantajlı")
            print(f"  {n_steps:>6} | {n_envs:>6} | {sa:>16.2f}x | {sb:>16.2f}x | {note:>26}")
    print("  " + "-" * 96)

    # --- otomatik yorum blogu ---
    print("\n  YORUM (yalnızca yukarıdaki sayılardan türetilmiştir):")
    for label in ("A (SmallStateParallelEnv)", "B (LargeStateSequentialEnv)"):
        vals = {(t, n): v for (l, t, n), v in summary.items() if l == label}
        if not vals:
            continue
        faster = sum(1 for v in vals.values() if v >= 1.0)
        best = max(vals.items(), key=lambda kv: kv[1])
        worst = min(vals.items(), key=lambda kv: kv[1])
        print(f"\n   Env {label}:")
        print(f"     - {len(vals)} kombinasyonun {faster}'inde JAX NumPy'dan hızlı.")
        print(f"     - en iyi: T={best[0][0]}, N={best[0][1]} -> {best[1]:.2f}x faster")
        if worst[1] >= 1.0:
            print(f"     - en kötü: T={worst[0][0]}, N={worst[0][1]} -> {worst[1]:.2f}x faster")
        else:
            print(f"     - en kötü: T={worst[0][0]}, N={worst[0][1]} -> "
                  f"{worst[1]:.2f}x / {1.0/worst[1]:.2f}x SLOWER")
        for n_steps in ROLLOUT_LENGTHS:
            seq = [(n, vals[(n_steps, n)]) for n in NUM_ENVS_LIST if (n_steps, n) in vals]
            if len(seq) >= 2:
                trend = "azalıyor" if seq[-1][1] < seq[0][1] else "artıyor"
                print(f"     - T={n_steps}: N {seq[0][0]}->{seq[-1][0]} giderken JAX avantajı "
                      f"{seq[0][1]:.2f}x -> {seq[-1][1]:.2f}x ({trend})")
        for n_envs in NUM_ENVS_LIST:
            pair = [(t, vals[(t, n_envs)]) for t in ROLLOUT_LENGTHS if (t, n_envs) in vals]
            if len(pair) == 2:
                trend = "artıyor" if pair[1][1] > pair[0][1] else "azalıyor"
                print(f"     - N={n_envs}: T {pair[0][0]}->{pair[1][0]} giderken JAX avantajı "
                      f"{pair[0][1]:.2f}x -> {pair[1][1]:.2f}x ({trend})")

    print("\n   NEDENSELLİK UYARISI: yukarıdaki eğilimlerin 'CPU doygunluğu', 'cache")
    print("   miss' veya 'memory bandwidth' ile açıklanması bu deneyde ÖLÇÜLMEMİŞTİR.")
    print("   Bunu iddia etmek için donanım sayaçlarıyla (perf / VTune / Nsight) profil")
    print("   almak gerekir. Buradaki tablolar yalnızca duvar saati süresi ve SPS")
    print("   gösterir; sebep DEĞİL, sonuç raporlarlar.")
    print("   Env B bu yüzden 'memory-bound' değil, 'large-state, scattered-access,")
    print("   sequentially dependent workload' olarak adlandırılmıştır.")


def report_rust_baseline():
    print("\n" + "-" * 100)
    print("  Opsiyonel Rust CPU baseline'ı (Environment B için)")
    print("-" * 100)
    cargo = shutil.which("cargo")
    rustc = shutil.which("rustc")
    if cargo is None and rustc is None:
        print("  Bu sistemde Rust toolchain'i (cargo/rustc) BULUNAMADI -> Rust baseline'ı atlandı.")
        print("  Ana deney (NumPy + JAX) bundan etkilenmez; sonuçların tamamı yukarıda.")
        print("  Karten et al.'ın 'Rust suits sequential or memory-intensive environments'")
        print("  iddiasının doğrudan ölçümü bu çalıştırmada YAPILMAMIŞTIR — aşağıdaki")
        print("  Environment B sonuçları yalnızca JAX tarafının ne yaptığını gösterir.")
    else:
        print(f"  Rust toolchain'i bulundu (cargo={cargo}, rustc={rustc}).")
        print("  Bu script bir Rust baseline'ı DERLEMEZ: doğrulanmamış bir Rust")
        print("  implementation'ı eklemek, karşılaştırmayı sağlamlaştırmak yerine")
        print("  güvenilirliğini düşürürdü. Rust baseline'ı istiyorsanız ayrı bir")
        print("  crate olarak eklenip Environment B'nin step fonksiyonu birebir")
        print("  taşınmalı ve aynı correctness kontrolünden geçirilmelidir.")


# ===========================================================================
if __name__ == "__main__":
    print("=" * 100)
    print(" DENEY 8: RL Rollout Sistemleri — jit / vmap / lax.scan Kompozisyonu")
    print("=" * 100)

    devices = get_available_devices()
    print(f"  Bulunan cihazlar : {list(devices.keys())}")
    print(f"  JAX sürümü       : {jax.__version__}")
    print(f"  Benchmark        : repeats={REPEATS}, hedef pencere ~{TARGET_SECONDS*1000:.0f} ms/repeat, "
          f"çağrı başına bütçe {MAX_SECONDS_PER_CALL:.1f} s")
    print(f"  Matris           : N ∈ {list(NUM_ENVS_LIST)}, T ∈ {list(ROLLOUT_LENGTHS)}")
    print("  Hipotez          : jit+vmap+scan küçük-state/yüksek-paralellik env'lerde çok")
    print("                     uygun olabilir; sequential/memory-intensive env'lerde kazanç küçülebilir.")

    # ---------------------------------------------------------------------
    # ÖNCE CORRECTNESS — geçmezse hiçbir performans sayısı basılmaz
    # ---------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  CORRECTNESS (performans ölçümlerinden ÖNCE)")
    print("=" * 100)
    ok_a = check_correctness(ENV_A, devices)
    ok_b = check_correctness(ENV_B, devices)

    if not (ok_a and ok_b):
        print("\n  HATA: implementation'lar aynı sonucu üretmiyor. Performans sonuçları")
        print("        anlamsız olacağı için ölçüm YAPILMADI.")
        raise SystemExit(1)

    print("\n  Tüm implementation'lar NumPy baseline ile eşleşiyor. Ölçüme geçiliyor.")

    results_a = run_env_benchmarks(ENV_A, devices)
    results_b = run_env_benchmarks(ENV_B, devices)

    print("\n" + "=" * 100)
    print("  ÖZET")
    print("=" * 100)
    summarize(ENV_A, results_a, devices)
    summarize(ENV_B, results_b, devices)
    compare_environments(results_a, results_b, devices)
    numpy_vs_jax_tables(results_a, results_b)
    report_rust_baseline()

    print("\n" + "=" * 100)
