# Image Compression Using Randomized SVD

**Matrix Theory (EE1030) - Course Project**
**Author-AI25BTECH11030-Sarvesh Tamgade**

## Project Structure

```
SoftwareAssignment/
├── codes/
│   |── c_main/
│   |    └── main.c
|   └──README_code.md
├── figs/ (compressed images)
├── tables/
│   └── table.tex
├── report.pdf
└── README.md

```

## Main Components in main.c

- Image I/O:.jpg read/write
- Matrix operations: Multiplication, transpose, norms, allocation
- QR decomposition: Modified Gram-Schmidt
- Eigendecomposition: QR iteration
- Randomized SVD: Main algorithm with power iterations
- Image reconstruction and compression
