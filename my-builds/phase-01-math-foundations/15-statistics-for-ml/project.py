"""Lesson 15: Statistics for ML - from scratch demos."""
import math
import numpy as np

rng = np.random.default_rng(7)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
section("1) Sample variance: divide by n-1 (Bessel), not n")

true_var = 4.0  # normal with sigma=2
n, trials = 5, 200000
samples = rng.normal(0, 2, (trials, n))
means = samples.mean(axis=1, keepdims=True)
ss = ((samples - means) ** 2).sum(axis=1)
var_n = (ss / n).mean()
var_n1 = (ss / (n - 1)).mean()
print(f"True variance: {true_var}. Samples of size n={n}, averaged over {trials} trials.")
print(f"  divide by n   : average estimate = {var_n:.4f}  <- too small (biased)")
print(f"  divide by n-1 : average estimate = {var_n1:.4f}  <- on target (unbiased)")
print("Why: the sample mean sits closer to its own data than the true mean does,")
print("so spread around it looks smaller. n-1 corrects for that stolen freedom.")
assert abs(var_n1 - true_var) < 0.05
assert abs(var_n - true_var * (n - 1) / n) < 0.05
assert abs(np.var(samples[0], ddof=1) - ss[0] / (n - 1)) < 1e-9  # matches numpy ddof=1

# ---------------------------------------------------------------
section("2) p-value from scratch: permutation test")

group_a = rng.normal(0.0, 1.0, 40)
group_b = rng.normal(0.5, 1.0, 40)  # real difference of 0.5
observed = group_b.mean() - group_a.mean()
print(f"Group A mean: {group_a.mean():.3f}, Group B mean: {group_b.mean():.3f}")
print(f"Observed difference: {observed:.3f}")
print("Question: could a difference this big happen by pure shuffling luck?")

combined = np.concatenate([group_a, group_b])
n_perm = 10000
count = 0
for _ in range(n_perm):
    rng.shuffle(combined)
    diff = combined[40:].mean() - combined[:40].mean()
    if abs(diff) >= abs(observed):
        count += 1
p_value = count / n_perm
print(f"Shuffled labels {n_perm} times. {count} shuffles matched or beat it.")
print(f"p-value = {p_value:.4f} -> shuffling luck almost never does this. Real effect.")
assert p_value < 0.05

null_a, null_b = rng.normal(0, 1, 40), rng.normal(0, 1, 40)
obs0 = null_b.mean() - null_a.mean()
comb0 = np.concatenate([null_a, null_b])
count0 = 0
for _ in range(n_perm):
    rng.shuffle(comb0)
    if abs(comb0[40:].mean() - comb0[:40].mean()) >= abs(obs0):
        count0 += 1
p0 = count0 / n_perm
print(f"\nControl: two groups from the SAME distribution -> p-value = {p0:.4f} (big, as it should be)")
assert p0 > 0.05

# ---------------------------------------------------------------
section("3) Multiple comparisons trap: 20 tests at alpha=0.05")

alpha, n_tests, experiments = 0.05, 20, 20000
# Each test on null data has a 5% false-positive chance. Run 20 per experiment.
p_vals = rng.uniform(0, 1, (experiments, n_tests))  # null p-values are uniform
any_false_pos = (p_vals < alpha).any(axis=1).mean()
theory = 1 - (1 - alpha) ** n_tests
print(f"Testing {n_tests} model configs where NONE is actually better (alpha={alpha}).")
print(f"  Theory: P(at least one 'significant' fluke) = 1 - 0.95^20 = {theory:.3f}")
print(f"  Simulated over {experiments} experiments:          {any_false_pos:.3f}")
print("Run enough tests and a 'winner' appears by luck alone ~64% of the time.")
assert abs(any_false_pos - theory) < 0.01

bonferroni = alpha / n_tests
any_fp_corrected = (p_vals < bonferroni).any(axis=1).mean()
print(f"  Bonferroni fix (use alpha/{n_tests} = {bonferroni:.4f}): rate drops to {any_fp_corrected:.3f}")
assert any_fp_corrected < 0.06

# ---------------------------------------------------------------
section("4) Statistically significant but tiny: huge n trap")


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def two_prop_z_test(acc_a, acc_b, n):
    pool = (acc_a + acc_b) / 2
    se = math.sqrt(2 * pool * (1 - pool) / n)
    z = (acc_b - acc_a) / se
    return z, 2 * (1 - norm_cdf(abs(z)))


acc_a, acc_b = 0.9234, 0.9237
for n_test in [1000, 100000, 10_000_000]:
    z, p = two_prop_z_test(acc_a, acc_b, n_test)
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"  n = {n_test:>10,}: same 0.03% gap -> p = {p:.4f} ({verdict})")
print("The gap never changed. Only n did. Huge n makes ANY tiny difference 'significant'.")
print("Significant means 'probably not zero', NOT 'big enough to care about'.")
_, p_small = two_prop_z_test(acc_a, acc_b, 1000)
_, p_big = two_prop_z_test(acc_a, acc_b, 10_000_000)
assert p_small > 0.05 and p_big < 0.05

# ---------------------------------------------------------------
section("5) Bootstrap confidence interval for comparing two 'models'")

n_examples = 2000
difficulty = rng.uniform(0, 1, n_examples)
model_a = (rng.uniform(0, 1, n_examples) > difficulty * 0.4).astype(float)
model_b = (rng.uniform(0, 1, n_examples) > difficulty * 0.36).astype(float)
obs_diff = model_b.mean() - model_a.mean()
print(f"Model A accuracy: {model_a.mean():.4f}, Model B accuracy: {model_b.mean():.4f}")
print(f"Observed difference (B - A): {obs_diff:+.4f}")

boots = 5000
idx = rng.integers(0, n_examples, (boots, n_examples))
diffs = model_b[idx].mean(axis=1) - model_a[idx].mean(axis=1)
diffs.sort()
lo, hi = diffs[int(0.025 * boots)], diffs[int(0.975 * boots)]
print(f"Bootstrap: resampled the SAME test examples (paired) {boots} times.")
print(f"95% CI for the difference: [{lo:+.4f}, {hi:+.4f}]")
if lo > 0 or hi < 0:
    print("CI excludes 0 -> the difference looks real on this test set.")
else:
    print("CI includes 0 -> cannot tell the models apart on this test set.")
assert lo < obs_diff < hi
assert diffs.std() < 0.02  # paired resampling keeps the spread tight

print("\nAll checks passed.")
