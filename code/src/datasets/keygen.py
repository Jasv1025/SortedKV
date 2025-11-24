import json
import math
import os
import random
import string
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
"""This is a script to generate a key dataset based on cdfs and a config file to emluate real world usage"
We found the SOSD benchmark to be too limited as it only gens int len keys, and YSCB did not offer
enough flexblity
A learned indexs preformance is dependent on  the datas distrbution. As such, this script
allows for generation of many different kinds of distrib.
Chosen distrbs: uniform for something uniform. the model should preform well.
inverted cdf: a learned index trys to approx a cdf. so to gen a target cdf, we can use inverted cdf to generate a key space
that approx follows the cdf
zipfian, pareto: two distrbs that mock data where a small % of key span is where most keys live. very realistic. zipfian we normalize ranks 

stocahstic: just random, for fun. 

we can compose multiple "componets" to make a dynamic dataset.
"""
import matplotlib.pyplot as plt
from scipy.stats import pareto, zipf

len_sample = Callable[[], int]
val_samp = Callable[..., float]
key_gen = Callable[..., str]

path = "config.json"
output_path = "generated_full_dataset.txt"
default_key_c = 10000
@dataclass(frozen=True)
class Component:
    name: str
    weight: float
    type: str
    namespace: str = ""
    sep: str = ":"
    generator: Optional[key_gen] = None
    config: Optional[Dict] = None

def charset_from_name(name: Optional[str]) -> str:
    name = (name or "").lower()
    if name in ("lower", "ascii_lowercase"):
        return string.ascii_lowercase
    if name in ("upper", "ascii_uppercase"):
        return string.ascii_uppercase
    if name in ("letters", "ascii_letters"):
        return string.ascii_letters
    if name in ("digits", "numeric"):
        return string.digits
    if name in ("alnum", "letters_digits"):
        return string.ascii_letters + string.digits
    if name in ("alnum_upper", "upper_digits"):
        return string.ascii_uppercase + string.digits
    if name in ("alnum_lower", "lower_digits"):
        return string.ascii_lowercase + string.digits
    return string.ascii_letters + string.digits


def prefix(namespace: str, sep: str, s: str) -> str:
    return (namespace + sep + s) if namespace else s


def create_uniform_length_sampler(min_len: int, max_len: int) -> len_sample:
    return lambda: random.randint(min_len, max_len)

"i did more thinking and this may not fully preserve lexiographically ordering in numeric form"
def map_number_to_base_string(value: int, length: int, charset: str) -> str:
    "base 26 :/. python 'ints' are a godsend"
    if value < 0:
        raise ValueError(f"Value cannot be negative: {value}")
    base = len(charset)
    max_value = base ** length - 1
    if value > max_value:
        value = max_value
    if value == 0:
        return charset[0] * length

    chars: List[str] = []
    temp = int(value)
    while len(chars) < length:
        temp, rem = divmod(temp, base)
        chars.append(charset[rem])
    return "".join(reversed(chars))


def generate_distributed_string(p: float, length_sampler: len_sample, charset: str) -> str:
    key_len = length_sampler()
    base = len(charset)
    space = base ** key_len
    target = int(p * (space - 1))
    return map_number_to_base_string(target, key_len, charset)

def generate_short_random() -> str:
    charset = string.ascii_lowercase
    key_len = random.randint(5, 10)
    return "".join(random.choice(charset) for _ in range(key_len))


def generate_sequential_id(i: int) -> str:
    base_id = 0
    return f"{base_id + i}"

def build_component(raw: Dict) -> Optional[Component]:
    t = raw.get("type")
    weight = float(raw.get("weight", 0.0))
    ns = raw.get("namespace", "")
    sep = raw.get("namespace_sep", ":")

    if t == "stochastic":
        return Component(
            name=raw.get("name", "Usernames"),
            weight=weight,
            type="stochastic",
            namespace=ns,
            sep=sep,
            generator=generate_short_random,
        )

    if t == "sequential_ids":
        return Component(
            name=raw.get("name", "Sequential IDs"),
            weight=weight,
            type="deterministic_sequence",
            namespace=ns,
            sep=sep,
            generator=generate_sequential_id,
        )

    if t in ("uniform", "pareto", "zipf"):
        cfg = raw.get("config", {})
        L = cfg.get("length", {"min": 8, "max": 12})
        charset = cfg.get("charset_custom") or charset_from_name(cfg.get("charset", "alnum"))
        length_sampler = create_uniform_length_sampler(int(L.get("min", 8)), int(L.get("max", 12)))

        value_sampler: Optional[Callable] = None
        gen_type: Optional[str] = None

        if t in ("uniform", "pareto"):
            gen_type = "normalized_inverse_cdf"
            if t == "uniform":
                value_sampler = lambda p: p
            else: # pareto
                b = float(cfg.get("b", 1.5))
                value_sampler = lambda p, bb=b: pareto.ppf(p, b=bb)
        elif t == "zipf":
            gen_type = "normalized_zipf"
            a = float(cfg.get("a", 1.3))
            value_sampler = lambda: int(zipf.rvs(a))

        if value_sampler and gen_type:
            return Component(
                name=raw.get("name", t.title()),
                weight=weight,
                type=gen_type,
                namespace=ns,
                sep=sep,
                config={
                    "value_sampler": value_sampler,
                    "length_sampler": length_sampler,
                    "charset": charset,
                },
            )

    return None

def load_model_components(path: str = path) -> Tuple[List[Component], int, str]:
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = json.load(f)

        components: List[Component] = []
        for c in cfg.get("components", []):
            built = build_component(c)
            if built:
                components.append(built)

        if components:
            return components, int(cfg.get("num_keys", default_key_c)), cfg.get("output", output_path)
    return

def generate_for_stochastic(comp: Component, n: int) -> List[str]:
    return [prefix(comp.namespace, comp.sep, comp.generator()) for _ in range(n)]


def generate_for_sequence(comp: Component, n: int) -> List[str]:
    return [prefix(comp.namespace, comp.sep, comp.generator(i)) for i in range(n)]


def normalize(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return [0.0 for _ in values]
    return [(v - lo) / span for v in values]


def generate_for_normalized_inverse_cdf(comp: Component, n: int) -> List[str]:
    cfg = comp.config or {}
    vs: Callable[[float], float] = cfg["value_sampler"]
    ls: len_sample = cfg["length_sampler"]
    charset: str = cfg["charset"]

    raw = [float(vs(random.random())) for _ in range(n)]
    probs = normalize(raw)

    return [
        prefix(comp.namespace, comp.sep, generate_distributed_string(p, ls, charset))
        for p in probs
    ]

#this is only partially correct, but it is fine for now
def generate_for_normalized_zipf(comp: Component, n: int) -> List[str]:
    cfg = comp.config or {}
    vs: Callable[[], int] = cfg["value_sampler"]
    ls: len_sample = cfg["length_sampler"]
    charset: str = cfg["charset"]

    ranks = [int(vs()) for _ in range(n)]
    rmin, rmax = min(ranks), max(ranks)
    span = (rmax - rmin) or 1
    probs = [(r - rmin) / span for r in ranks]

    return [
        prefix(comp.namespace, comp.sep, generate_distributed_string(p, ls, charset))
        for p in probs
    ]


def generate_keys_for_component(comp: Component, n: int) -> List[str]:
    if n <= 0:
        return []

    gen_map = {
        "stochastic": generate_for_stochastic,
        "deterministic_sequence": generate_for_sequence,
        "normalized_inverse_cdf": generate_for_normalized_inverse_cdf,
        "normalized_zipf": generate_for_normalized_zipf,
    }
    if generator_func := gen_map.get(comp.type):
        return generator_func(comp, n)
    return []

def compute_component_counts(components: List[Component], total: int) -> List[Tuple[Component, int]]:
    weight_sum = sum(c.weight for c in components)
    if not math.isclose(weight_sum, 1.0):
        print(f"Warning: Component weights sum to {weight_sum:.6f}, not 1.0.")

    counts = [(c, int(total * c.weight)) for c in components]
    return counts


def generate_all_keys(components: List[Component], total: int) -> List[str]:
    per_component = compute_component_counts(components, total)

    all_keys: List[str] = []
    for comp, n in per_component:
        print(f"  Generating {n:>6} for '{comp.name}' ({comp.type})...")
        all_keys.extend(generate_keys_for_component(comp, n))

    random.shuffle(all_keys)
    return all_keys


def write_lines(path: str, lines: Iterable[str]) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return os.path.abspath(path)


def sample_print(keys: List[str], k: int = 10) -> None:
    k = min(k, len(keys))
    if k == 0:
        print("(no keys)")
        return
    print("\n--- Sample of Generated Keys ---")
    for s in random.sample(keys, k):
        print(s)
    print("...")
def main(config_path: str = path) -> None:
    components, num_keys, out_path = load_model_components(config_path)
    all_keys = generate_all_keys(components, num_keys)
    real_n = len(all_keys)
    print(f"Writing {real_n} keys to '{out_path}'...")
    if real_n != num_keys:
        print(f"Note: generated {real_n} instead of {num_keys} due to improper fractions\n ")
    write_lines(out_path, all_keys)

    sample_print(all_keys, k=10)

if __name__ == "__main__":
    main()