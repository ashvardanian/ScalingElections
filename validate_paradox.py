"""
Minimal self-contained demonstration of the participation paradox in Schulze voting.

This script implements the complete Schulze method (pairwise preferences + Floyd-Warshall
for strongest paths) and validates a counterintuitive edge case: adding a voter who ranks
Java first causes Java to lose and Python to win instead. Requires only NumPy.
"""
import numpy as np

def pairwise_preferences(rankings):
    n = max(np.max(r) for r in rankings) + 1; prefs = np.zeros((n, n), dtype=np.uint32)
    [[prefs.__setitem__((p, o), prefs[p, o] + 1) for i, p in enumerate(r) for o in r[i+1:]] for r in rankings]
    return prefs

def widest_paths(prefs):
    n, p = prefs.shape[0], np.zeros((prefs.shape[0], prefs.shape[0]), dtype=np.uint32)
    [[p.__setitem__((i, j), prefs[i, j]) for i in range(n) for j in range(n) if i != j and prefs[i, j] > prefs[j, i]]]
    [[[p.__setitem__((j, k), max(p[j, k], min(p[j, i], p[i, k]))) for k in range(n) if i != k and j != k] for j in range(n) if i != j] for i in range(n)]
    return p

def winner(rankings):
    p = widest_paths(pairwise_preferences(rankings))
    wins = [(p[i] > p[:,i]).sum() for i in range(4)]
    return ['Python','Rust','Go','Java'][np.argmax(wins)], max(wins)

r = [np.array([0,3,1,2])] + [np.array([0,3,2,1])]*2 + [np.array([1,0,2,3])]*2 + [np.array([2,3,1,0])]*4 + [np.array([3,0,1,2])]
i_w, i_n = winner(r); r.append(np.array([3,0,2,1])); f_w, f_n = winner(r)
print(f"Initial (10 voters): {i_w} wins ({i_n} victories)\nAfter adding Java-first voter: {f_w} wins ({f_n} victories)\n{'✓ Paradox confirmed!' if i_w != f_w else '✗ No paradox'}")
