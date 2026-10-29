"""Identity round trip SMEM -> tcgen05_copy -> TMEM for the upstream M=128, N=64 FP32 example (10-tcgen05-copy.py L116-L157)."""
def reference(X):
    return X.clone()
