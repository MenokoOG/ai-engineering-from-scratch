"""Lesson 07: Bayes' Theorem - from scratch demos."""
import math
from collections import Counter


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
section("1) Rare disease test: 99% accurate, but posterior is small")


def posterior_disease(prevalence, sensitivity, specificity):
    p_pos_given_sick = sensitivity
    p_pos_given_healthy = 1 - specificity
    p_pos = p_pos_given_sick * prevalence + p_pos_given_healthy * (1 - prevalence)
    return p_pos_given_sick * prevalence / p_pos


prevalence = 1 / 10000
post = posterior_disease(prevalence, sensitivity=0.99, specificity=0.99)
print(f"Prevalence (prior): 1 in 10,000 = {prevalence:.4%}")
print("Test: 99% sensitive (catches sick), 99% specific (clears healthy)")
print(f"P(sick | positive test) = {post:.4%}  <- under 1%!")
assert abs(post - 0.0098) < 0.0005

print("\nSame thing with whole-number counts (1,000,000 people):")
n = 1_000_000
sick = n * prevalence
true_pos = sick * 0.99
false_pos = (n - sick) * 0.01
print(f"  sick people: {sick:.0f}, of whom {true_pos:.1f} test positive")
print(f"  healthy people wrongly positive: {false_pos:.1f}")
print(f"  so a positive means: {true_pos:.1f} / {true_pos + false_pos:.1f} = {true_pos/(true_pos+false_pos):.4%}")
print("False positives from the huge healthy crowd swamp the true positives.")

post2 = posterior_disease(1 / 100, 0.99, 0.99)
print(f"\nIf the disease were 1 in 100 instead: posterior = {post2:.2%}. Priors matter.")

# ---------------------------------------------------------------
section("2) Naive Bayes spam classifier with Laplace smoothing")

train = [
    ("win money now", "spam"),
    ("win free prize now", "spam"),
    ("claim free money", "spam"),
    ("meeting notes attached", "ham"),
    ("lunch meeting tomorrow", "ham"),
    ("project notes for tomorrow", "ham"),
]


def train_nb(data, alpha):
    class_counts = Counter(label for _, label in data)
    word_counts = {c: Counter() for c in class_counts}
    vocab = set()
    for text, label in data:
        for w in text.split():
            word_counts[label][w] += 1
            vocab.add(w)
    return class_counts, word_counts, vocab, alpha, len(data)


def log_posteriors(model, text):
    class_counts, word_counts, vocab, alpha, n_docs = model
    scores = {}
    for c in class_counts:
        score = math.log(class_counts[c] / n_docs)
        total_words = sum(word_counts[c].values())
        for w in text.split():
            if w not in vocab:
                continue
            p = (word_counts[c][w] + alpha) / (total_words + alpha * len(vocab))
            if p == 0.0:
                score = -math.inf
                break
            score += math.log(p)
        scores[c] = score
    return scores


def classify(model, text):
    scores = log_posteriors(model, text)
    return max(scores, key=scores.get), scores


test_msg = "free money meeting"
print(f"Training on {len(train)} toy emails. Test message: '{test_msg}'")

no_smooth = train_nb(train, alpha=0.0)
_, scores0 = classify(no_smooth, test_msg)
print("\nWithout smoothing (alpha=0):")
print(f"  spam score: {scores0['spam']}, ham score: {scores0['ham']}")
print("  'meeting' never appeared in spam and 'money' never in ham,")
print("  so ONE unseen word zeroes out a whole class. Both scores are -inf. Useless.")
assert scores0["spam"] == -math.inf and scores0["ham"] == -math.inf

smooth = train_nb(train, alpha=1.0)
label, scores1 = classify(smooth, test_msg)
print("\nWith Laplace smoothing (alpha=1, pretend every word was seen once extra):")
print(f"  spam log-score: {scores1['spam']:.3f}, ham log-score: {scores1['ham']:.3f}")
print(f"  prediction: {label}")
assert label == "spam"
assert classify(smooth, "lunch notes tomorrow")[0] == "ham"
print("  'lunch notes tomorrow' -> ham (checked)")

# ---------------------------------------------------------------
section("3) MLE vs MAP coin estimate")

heads, flips = 3, 3
mle = heads / flips
print(f"Data: {heads} heads in {flips} flips (small sample!)")
print(f"MLE (just the data): p = {heads}/{flips} = {mle:.2f}  <- claims the coin NEVER lands tails")

a, b = 2, 2  # Beta(2,2) prior: gentle belief that coins are fair-ish
map_est = (heads + a - 1) / (flips + a - 1 + b - 1)
print(f"MAP with Beta({a},{b}) prior: p = ({heads}+{a-1})/({flips}+{a-1+b-1}) = {map_est:.2f}")
print("The prior acts like 2 imaginary flips (1 head, 1 tail) pulling toward 0.5.")
assert mle == 1.0
assert abs(map_est - 0.8) < 1e-12

big_heads, big_flips = 600, 1000
big_mle = big_heads / big_flips
big_map = (big_heads + a - 1) / (big_flips + a - 1 + b - 1)
print(f"\nWith lots of data ({big_heads}/{big_flips}): MLE={big_mle:.4f}, MAP={big_map:.4f}")
print("Given enough data, the prior barely matters. MLE and MAP converge.")
assert abs(big_mle - big_map) < 0.001

# ---------------------------------------------------------------
section("4) Sequential Bayesian updating: Beta(1,1) + 7H 3T -> Beta(8,4)")

a, b = 1, 1  # Beta(1,1) = flat prior, "no idea"
observations = ["H"] * 7 + ["T"] * 3
print("Start: Beta(1,1), a flat prior over the coin's heads-probability.")
print("Update rule: heads -> a+1, tails -> b+1. One observation at a time:")
for i, obs in enumerate(observations, 1):
    if obs == "H":
        a += 1
    else:
        b += 1
    mean = a / (a + b)
    print(f"  after obs {i:2d} ({obs}): Beta({a},{b}), mean estimate = {mean:.3f}")

assert (a, b) == (8, 4)
assert abs(a / (a + b) - 2 / 3) < 1e-12
mode = (a - 1) / (a + b - 2)
print(f"\nFinal posterior: Beta({a},{b}). Mean = {a}/{a+b} = {a/(a+b):.3f}, mode = {mode:.3f} (the MLE, 7/10)")
print("Order does not matter: all at once or one by one gives the same posterior.")

print("\nAll checks passed.")
