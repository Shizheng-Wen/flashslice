# Numerics and determinism

## The gate

The fused path is not held to a tolerance. It is held to eager: for every
output and every parameter gradient, the fused result must sit as close to an
fp64 reference as eager fp32 does, within 1.25× (and within 3× where eager
itself sits above 1e-4 from the truth, i.e. where the quantity is
cancellation noise in both implementations). bf16-dot modes are gated at 2× of
eager-under-autocast's error, tf32-class modes at an order of magnitude. Two
fused runs must agree bitwise. `bench/parity_test.py` applies this to the
layer at the paper's shapes, `bench/parity_ops.py` to the ops on every shape,
weight layout and call order the layer does not exercise.

## The canary: the temperature gradient

\[
\frac{\partial L}{\partial \tau} = \sum_n \sum_g dl_{ng}\left(-\frac{\text{logit}_{ng}}{\tau}\right),
\qquad dl_{ng} = w_{ng}\,(dw_{ng} - \delta_n),\qquad \delta_n = \sum_g w_{ng}\, dw_{ng}.
\]

In exact arithmetic \(\sum_g dl_{ng} = \delta_n\,(1 - \sum_g w_{ng}) = 0\) on
every row. Anything that breaks that identity *coherently* across a row — a
row of \(w\) that does not sum to one, or a \(\delta_n\) formed from
differently rounded products than the \(dl\) — becomes a bias scaled by the
logits, while every other gradient shrugs it off. This gradient failed the
gate three separate times while the blocked kernels were built, and each time
it was the only one that did. Three rules came out of it:

1. **Save \((m, l)\), not \(\text{lse} = m + \log l\).** Recomputing
   \(\exp(\text{logit} - \text{lse})\) rounds the argument of the dominant
   weights.
2. **The backward kernels that own points normalize with a row sum they form
   themselves**, from the very exponentials they use, so a row of \(w\) sums
   to one to the precision of one summation whatever rounding the saved
   \(l\) carries (`set_own_row_sum` switches this off for attribution). The
   saved \(l\) comes from an online pass, rescaled once per move of the row
   max. Outputs and the slot-owning kernels take the saved \(l\): for them a
   row factor of \(1 + O(\text{ulp})\) is a relative perturbation of that
   row's contribution, not a cancellation.
3. **\(\delta_n\) and \(dl_{ng}\) are built from the same rounded products
   \(e_{ng}\, dw_{ng}\)**: \(\delta_n = (\sum_g e_{ng} dw_{ng}) / l_n\) in the
   first pass and \(dl_{ng} = (e_{ng} dw_{ng} - e_{ng} \delta_n) / l_n\) in the
   second. Normalizing first and forming \(\delta_n\) from \(w_{ng} dw_{ng}\)
   while \(dl\) uses \(w_{ng}(dw_{ng} - \delta_n)\) puts a second,
   independently rounded product into the identity and cost 1.5–1.9× eager on
   \(d\tau\); the algebraically equal \(f_n \cdot df_n + w_n \cdot ds\) (one
   dot cheaper) cost 3×.

And the temperature gradient itself is summed **row by row over all of
\(G\)** before anything is added across rows, in the point-owning kernel.
Summing each block's share over the points first and cancelling across blocks
at the end put 2–3× eager's error on it.

## Precision levels

| level | logits dot | value dots | inputs | gate |
| --- | --- | --- | --- | --- |
| `ieee` | fp32 FMA | fp32 FMA | fp32 or bf16 | 1.25× eager fp32 |
| `tf32` | fp32 FMA | tf32 tensor cores | fp32 or bf16 | 10× (order of magnitude) |
| `bf16v` | fp32 FMA | bf16 tensor cores | bf16 | 2× eager bf16 autocast |
| `bf16` | bf16 tensor cores | bf16 tensor cores | bf16 | 2× eager bf16 autocast |
| `tf32x3` | three tf32 products | three tf32 products | fp32 | 10×; measured 3–7× eager fp32 |

The logits dot stays fp32 below the `bf16` level because the softmax Jacobian
amplifies noise in the slice weights. `tf32x3` was the attempt to get fp32
accuracy on tensor cores; Triton's implementation lands two to three bits
short of it. In a training run at large \(G\) with bf16 activations, `bf16`
is the mode to use: it matched a tf32 control at every step of a 50k-step
run on the coupling of a point-cloud model, at 3.5× less step time.

## Determinism

No kernel uses atomics. Slot-owning quantities are written as per-program
partials into a buffer of fixed layout and reduced with one `.sum()` over the
partition axis, in a fixed order. Two runs of the fused path produce bitwise
identical outputs and gradients; the parity gate checks it on every case. This
is a property the eager path does not have when it goes through
`index_add_` or scatter kernels, and it is what makes a fused-vs-eager
discrepancy attributable: if the fused path is deterministic and eager is
not, the noise is eager's.

## What blurs a comparison

Two things have been mistaken for kernel error and were not:

- **TF32 by container.** NGC PyTorch images set
  `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, and then every fp32 cuBLAS matmul
  runs in TF32 whatever `allow_tf32` says. Eager's einsums are then the
  imprecise side (≈3e-4 from fp64) while the `ieee` kernels sit at 1e-6 to
  1e-7. Compare both against fp64, or unset the variable.
- **Skinny cuBLAS shapes stay fp32.** The \(D = 32\) einsums of the paper's
  layer were kept in fp32 by cuBLAS even under that override, which is why
  their parity was bitwise where wider shapes showed 6e-4.

A model that reduces with scatter atomics elsewhere (a graph encoder, say) is
not repeatable run to run, and the kernels cannot be compared against it more
tightly than it agrees with itself. Measure eager's own repeat noise first.
