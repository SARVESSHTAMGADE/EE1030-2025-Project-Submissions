**Matrix Theory (EE1030) - Course Project**
**Author-AI25BTECH11030-Sarvesh Tamgade**
 Implementation: Randomized SVD for Image Compression

## Algorithm

**Randomized Singular Value Decomposition with Power Iterations**

Input: Matrix A (t × m), rank k, oversampling p, iterations q
Output: U, Σ, V for low-rank approximation

```
Y = A * Ω                           // Random projection (Ω: m × ℓ, ℓ = k+p)

for i = 1 to q:
    Y ← A * (A^T * Y)              // Power iteration
    Y ← QR_orthonormalize(Y)       // Orthonormalization

Q, R ← QR(Y)                        // QR decomposition

B ← Q^T * A                         // Project to smaller space

U_tilde, Σ, V ← SVD(B)             // SVD of smaller matrix

U ← Q * U_tilde                     // Recover U

return U_k, Σ_k, V_k               // Keep first k components
```

## Implementation Steps

1. Read grayscale image (PGM format)
2. Create random Gaussian matrix Ω
3. Compute initial sketch Y = AΩ
4. Apply q power iterations: Y ← A(A^T Y)
5. QR decomposition of Y using Modified Gram-Schmidt
6. Project: B = Q^T A
7. Eigendecomposition of BB^T for SVD
8. Reconstruct: A_k = U_k Σ_k V_k^T
9. Write compressed image (PGM format)

## Compilation

```bash
gcc -O3 -o compress codes/c_main/main.c -lm
```

## Running

```bash
./compress
```

Then provide:
- Input PGM filename
- Output PGM filename
- Rank k
