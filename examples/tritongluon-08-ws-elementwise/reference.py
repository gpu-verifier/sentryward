"""Dense FP32 addition for the upstream 1000 by 2000 warp-specialized test (08-warp-specialization.py L123-L320)."""
def reference(A, B):
    return A + B
