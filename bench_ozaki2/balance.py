#!/usr/bin/env python3
"""Break-even conditions for an Ozaki-II INT8 emulation of the p=255 volume GEMM.

native is compute bound (84% of the FP64 peak), the emulation is DRAM bound:

    t_nat = W / P_fp64            W = 2*Nq^4
    t_emu = T / B                 T = (9s + 8) * Nq^3

T is the traffic the library-GEMM structure cannot avoid:
    s INT8 GEMMs : write 4s*Nq^3 (INT32 C) + read s*Nq^3 (INT8 B)
    reconstruct  : read  4s*Nq^3            + write 8*Nq^3

so  t_emu < t_nat  <=>  P_fp64 / B  <  W/T  =  2*Nq / (9s + 8).
"""

S = 14                      # moduli needed for FP64-equivalent accuracy
NQ = 256                    # p = 255

W = lambda nq: 2.0 * nq ** 4
T = lambda nq, s=S: (9 * s + 8) * float(nq) ** 3
need = lambda nq, s=S: W(nq) / T(nq, s)          # FLOP/byte

MACHINES = [
    ("H100  FP64 TC   peak", 67.0e12, 3.35e12),
    ("H100  FP64 CUDA peak", 34.0e12, 3.35e12),
    ("GB200           peak", 40.1e12, 7.90e12),
    ("GB200  achieved (native 33.7 TF, volume_flux 7.09 TB/s)", 33.7e12, 7.09e12),
    ("GB200  achieved (native 33.7 TF, emulation  4.63 TB/s)", 33.7e12, 4.63e12),
]

print(f"traffic at Nq={NQ}, s={S}: {T(NQ)/1e9:.2f} GB  "
      f"({T(NQ)/(16.0*NQ**3):.1f}x the native {16.0*NQ**3/1e6:.0f} MB)\n")

print("condition 1 (arithmetic):  P_int8 / P_fp64 > s = %d" % S)
print("   GB200 measured ceiling 4724 TOP/s / 40.1 TFLOP/s = 117.8x"
      "  -> satisfied with %.1fx margin\n" % (117.8 / S))

print("condition 2 (balance):  P_fp64 / B  <  2*Nq/(9s+8)   [FLOP/byte]")
print("   s=14  ->  Nq/67\n")
print("      p      Nq   required FLOP/B")
for nq in (8, 128, 256, 340, 512, 640, 1024):
    print(f"   {nq-1:6d} {nq:7d} {need(nq):15.2f}")

print("\n   actual machine balance:")
for name, p, b in MACHINES:
    print(f"      {name:56s} {p/b:6.2f} FLOP/B  -> short by {p/b/need(NQ):.2f}x")

print("\n   Nq that would break even at each balance:")
for name, p, b in MACHINES[2:]:
    nq = p / b * (9 * S + 8) / 2
    print(f"      {name:56s} Nq = {nq:5.0f}  (p = {nq-1:.0f})")

print(f"\n   sensitivity to s at Nq={NQ}:  required = {2*NQ}/(9s+8)")
for s in (6, 8, 10, 12, 14, 18):
    print(f"      s={s:2d}: {need(NQ, s):5.2f} FLOP/B")

print("\nequivalently, at fixed hardware (Nq=%d, s=%d):" % (NQ, S))
for r, lab in ((1.0, "break-even"), (1 / 1.5, "1.5x win"), (0.5, "2x win")):
    print(f"   {lab:11s}: B > {T(NQ)*40.1e12/(W(NQ)*r)/1e12:5.1f} TB/s at 40.1 TFLOP/s"
          f"   |   P_fp64 < {W(NQ)*7.9e12*r/T(NQ)/1e12:5.1f} TFLOP/s at 7.9 TB/s")
