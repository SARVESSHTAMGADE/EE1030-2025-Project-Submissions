#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define M_PI 3.14159265358979323846
#define P_DEF 30
#define Q_DEF 6
#define QR_MAX 200
#define TOL 1e-9

int read_next_token(FILE *fp, char *buf, int bufsz) {
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') {
            while ((c = fgetc(fp)) != EOF && c != '\n') { }
            continue;
        }
        ungetc(c, fp);
        if (fscanf(fp, "%s", buf) == 1) return 1;
        return 0;
    }
    return 0;
}

float *read_pgm(const char *filename, int *width, int *height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror("fopen"); return NULL; }

    char fmt[16];
    if (!read_next_token(fp, fmt, sizeof(fmt))) { fclose(fp); return NULL; }
    if (!(fmt[0] == 'P' && fmt[1] == '5')) {
        printf("Only P5 supported (got %s)\n", fmt);
        fclose(fp);
        return NULL;
    }

    char token[128];
    if (!read_next_token(fp, token, sizeof(token))) { fclose(fp); return NULL; }
    int w = atoi(token);

    if (!read_next_token(fp, token, sizeof(token))) { fclose(fp); return NULL; }
    int h = atoi(token);

    if (!read_next_token(fp, token, sizeof(token))) { fclose(fp); return NULL; }
    int maxv = atoi(token);

    int c = fgetc(fp);
    if (c == '\r') {
        int c2 = fgetc(fp);
        if (c2 != '\n') ungetc(c2, fp);
    } else if (c != '\n' && !isspace(c)) {
        ungetc(c, fp);
    }

    if (w <= 0 || h <= 0) { fclose(fp); return NULL; }
    size_t size = (size_t)w * (size_t)h;

    unsigned char *buf = malloc(size);
    if (!buf) { fclose(fp); printf("malloc failed\n"); return NULL; }
    size_t r = fread(buf, 1, size, fp);
    fclose(fp);

    if (r != size) { printf("fread mismatch: got %zu expected %zu\n", r, size); free(buf); return NULL; }

    float *img = malloc(size * sizeof(float));
    if (!img) { free(buf); printf("malloc fail (img)\n"); return NULL; }

    float scale = 1.0f / (float)(maxv <= 0 ? 255 : maxv);
    for (size_t i = 0; i < size; i++) img[i] = ((float)buf[i]) * scale;

    free(buf);

    *width = w;
    *height = h;
    return img;
}

int write_pgm(const char *fn, const float *img, int w, int h) {
    FILE *f = fopen(fn, "wb");
    if (!f) return 1;

    fprintf(f, "P5\n%d %d\n255\n", w, h);

    size_t n = (size_t)w * (size_t)h;
    for (size_t i = 0; i < n; i++) {
        float x = img[i];
        if (x < 0) x = 0;
        if (x > 1) x = 1;
        unsigned char v = (unsigned char)(x * 255.0f + 0.5f);
        fwrite(&v, 1, 1, f);
    }

    fclose(f);
    return 0;
}

double randn_seeded(void) {
    static int have = 0;
    static double z1;

    if (have) { have = 0; return z1; }
    have = 1;

    double u1, u2;
    do { u1 = rand() / (RAND_MAX + 1.0); } while (u1 <= 1e-15);
    u2 = rand() / (RAND_MAX + 1.0);

    double r = sqrt(-2.0 * log(u1));
    double th = 2.0 * M_PI * u2;

    z1 = r * sin(th);
    return r * cos(th);
}

// matrix(rows x cols)
double **mat(int rows, int cols) {
    double **m = malloc(rows * sizeof(double*));
    if (!m) return NULL;

    for (int i = 0; i < rows; i++) {
        m[i] = calloc(cols, sizeof(double));
        if (!m[i]) {
            for (int j = 0; j < i; j++) free(m[j]);
            free(m);
            return NULL;
        }
    }

    return m;
}

// matrix(rows x cols)
void free_mat(int rows, double **m) {
    if (!m) return;
    for (int i = 0; i < rows; i++) free(m[i]);
    free(m);
}

double dot_d(int n, const double *a, const double *b) {
    double s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

double norm_d(int n, const double *a) {
    double s = dot_d(n, a, a);
    return s <= 0.0 ? 0.0 : sqrt(s);
}

// A (n x p), B (p x m), C (n x m)
void matmul(int n, int p, int m, double **A, double **B, double **C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            double s = 0.0;
            for (int r = 0; r < p; r++) s += A[i][r] * B[r][j];
            C[i][j] = s;
        }
}

// Y (n x l), Q (n x l), R (l x l)
void thin_mgs_qr(int n, int l, double **Y, double **Q, double **R) {
    for (int i = 0; i < l; i++)
        for (int j = 0; j < l; j++) R[i][j] = 0.0;

    for (int j = 0; j < l; j++) {
        for (int i = 0; i < n; i++) Q[i][j] = Y[i][j];

        for (int k = 0; k < j; k++) {
            double r = 0;
            for (int i = 0; i < n; i++) r += Q[i][k] * Q[i][j];
            R[k][j] = r;
            for (int i = 0; i < n; i++) Q[i][j] -= r * Q[i][k];
        }

        double rjj = 0;
        for (int i = 0; i < n; i++) rjj += Q[i][j] * Q[i][j];
        rjj = (rjj < 1e-14) ? 0.0 : sqrt(rjj);
        R[j][j] = rjj;

        if (rjj > 0.0)
            for (int i = 0; i < n; i++) Q[i][j] /= rjj;
        else
            for (int i = 0; i < n; i++) Q[i][j] = 0.0;
    }
}

// a (n x n), b (n x n), evecs (n x n)
int gram_step_full(int n, double **a, double **b, double **evecs) {
    double **Q = mat(n, n);
    double **R = mat(n, n);
    if (!Q || !R) { free_mat(n, Q); free_mat(n, R); return -1; }

    double *v = malloc(n * sizeof(double));
    double *qcol = malloc(n * sizeof(double));
    if (!v || !qcol) { free(v); free(qcol); free_mat(n, Q); free_mat(n, R); return -1; }

    for (int i = 0; i < n; i++) {
        for (int r = 0; r < n; r++) v[r] = a[r][i];

        for (int j = 0; j < i; j++) {
            for (int r = 0; r < n; r++) qcol[r] = Q[r][j];
            double rij = 0;
            for (int r = 0; r < n; r++) rij += qcol[r] * v[r];
            R[j][i] = rij;
            for (int r = 0; r < n; r++) v[r] -= qcol[r] * rij;
        }

        double nm = norm_d(n, v);
        R[i][i] = nm;

        if (nm > 1e-16)
            for (int r = 0; r < n; r++) Q[r][i] = v[r] / nm;
        else
            for (int r = 0; r < n; r++) Q[r][i] = 0.0;
    }

    matmul(n, n, n, R, Q, b);

    double **tmp = mat(n, n);
    if (!tmp) { free_mat(n, Q); free_mat(n, R); free(v); free(qcol); return -1; }
    matmul(n, n, n, evecs, Q, tmp);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            evecs[i][j] = tmp[i][j];

    free_mat(n, Q);
    free_mat(n, R);
    free_mat(n, tmp);
    free(v);
    free(qcol);

    return 0;
}

// A (n x n), B (n x n)
double frob_diff(int n, double **A, double **B) {
    double s = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double d = A[i][j] - B[i][j];
            s += d * d;
        }
    return sqrt(s);
}

// a (n x n), evecs (n x n)
int qr_eigendecomp(int n, double **a, double **evecs, double *eigvals) {
    double **An = mat(n, n);
    if (!An) return -1;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            evecs[i][j] = (i == j) ? 1.0 : 0.0;

    clock_t st = clock();

    for (int iter = 0; iter < QR_MAX; iter++) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                An[i][j] = 0.0;

        if (gram_step_full(n, a, An, evecs) != 0) { free_mat(n, An); return -1; }

        double diff = frob_diff(n, An, a);
        double elapsed = (double)(clock() - st) / CLOCKS_PER_SEC;

        if (iter % 20 == 0)
            printf("eig QR iter %4d diff=%.3e elapsed=%.2fs\n", iter, diff, elapsed);

        if (diff <= TOL) {
            for (int i = 0; i < n; i++) eigvals[i] = An[i][i];
            free_mat(n, An);
            printf("eig qr converged after %d iters\n", iter);
            return 0;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                a[i][j] = An[i][j];
    }

    for (int i = 0; i < n; i++) eigvals[i] = An[i][i];
    free_mat(n, An);
    printf("eig qr reached max iters\n");
    return 0;
}

// A (t x m), Omega (m x l), Y (t x l), tmp_m_l (m x l), Qtmp (t x l), Rtmp (l x l),
// Q (t x l), R (l x l), B (l x m), BBt (l x l), Evecs (l x l), U (t x k), V (m x k)
void randomized_svd(int t, int m, double **A, int k, int p, int q,
                    double ***U_out, double **S_out, double ***V_out)
{
    int maxnm = (t < m) ? t : m;
    int l = k + p;
    if (l > maxnm) l = maxnm;
    if (l < 1) l = 1;

    double **Omega = mat(m, l);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < l; j++)
            Omega[i][j] = randn_seeded();

    double **Y = mat(t, l);
    for (int i = 0; i < t; i++)
        for (int j = 0; j < l; j++) {
            double s = 0;
            for (int r = 0; r < m; r++) s += A[i][r] * Omega[r][j];
            Y[i][j] = s;
        }

    double **tmp_m_l = mat(m, l);
    double **Qtmp = mat(t, l);
    double **Rtmp = mat(l, l);

    for (int it = 0; it < q; it++) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < l; j++) tmp_m_l[i][j] = 0.0;

        for (int i = 0; i < m; i++)
            for (int r = 0; r < t; r++) {
                double a_r_i = A[r][i];
                for (int j = 0; j < l; j++) tmp_m_l[i][j] += a_r_i * Y[r][j];
            }

        for (int i = 0; i < t; i++)
            for (int j = 0; j < l; j++) Y[i][j] = 0.0;

        for (int i = 0; i < t; i++)
            for (int r = 0; r < m; r++) {
                double a_i_r = A[i][r];
                for (int j = 0; j < l; j++) Y[i][j] += a_i_r * tmp_m_l[r][j];
            }

        thin_mgs_qr(t, l, Y, Qtmp, Rtmp);

        for (int i = 0; i < t; i++)
            for (int j = 0; j < l; j++) Y[i][j] = Qtmp[i][j];

        printf("power iter %d/%d done\n", it + 1, q);
    }

    double **Q = mat(t, l);
    double **R = mat(l, l);
    thin_mgs_qr(t, l, Y, Q, R);

    double **B = mat(l, m);
    for (int i = 0; i < l; i++)
        for (int j = 0; j < m; j++) {
            double s = 0;
            for (int r = 0; r < t; r++) s += Q[r][i] * A[r][j];
            B[i][j] = s;
        }

    double **BBt = mat(l, l);
    for (int i = 0; i < l; i++)
        for (int j = 0; j < l; j++) {
            double s = 0;
            for (int r = 0; r < m; r++) s += B[i][r] * B[j][r];
            BBt[i][j] = s;
        }

    double **Evecs = mat(l, l);
    double *Eval = malloc(l * sizeof(double));
    qr_eigendecomp(l, BBt, Evecs, Eval);

    for (int i = 0; i < l; i++) {
        int maxi = i;
        for (int j = i + 1; j < l; j++) if (Eval[j] > Eval[maxi]) maxi = j;
        if (maxi != i) {
            double tv = Eval[i];
            Eval[i] = Eval[maxi];
            Eval[maxi] = tv;
            for (int r = 0; r < l; r++) {
                double t2 = Evecs[r][i];
                Evecs[r][i] = Evecs[r][maxi];
                Evecs[r][maxi] = t2;
            }
        }
    }

    double *S = malloc(k * sizeof(double));
    for (int i = 0; i < k; i++) {
        double v = Eval[i];
        if (v < 0) v = 0;
        S[i] = sqrt(v);
    }

    double **U = mat(t, k);
    for (int i = 0; i < t; i++)
        for (int j = 0; j < k; j++) {
            double s = 0;
            for (int r = 0; r < l; r++) s += Q[i][r] * Evecs[r][j];
            U[i][j] = s;
        }

    double **V = mat(m, k);
    for (int j = 0; j < k; j++) {
        double sigma = S[j];
        if (sigma < 1e-15) sigma = 1e-15;
        for (int i = 0; i < m; i++) {
            double s = 0;
            for (int r = 0; r < l; r++) s += B[r][i] * Evecs[r][j];
            V[i][j] = s / sigma;
        }
    }

    free_mat(m, Omega);
    free_mat(t, Y);
    free_mat(m, tmp_m_l);
    free_mat(t, Qtmp);
    free_mat(l, Rtmp);

    free_mat(t, Q);
    free_mat(l, R);
    free_mat(l, B);
    free_mat(l, BBt);
    free_mat(l, Evecs);

    free(Eval);

    *U_out = U;
    *S_out = S;
    *V_out = V;
}

// U (t x k), V (m x k)
float *reconstruct_from_usv(int t, int m, double **U, double *S, double **V, int k) {
    size_t tot = (size_t)t * (size_t)m;
    float *out = calloc(tot, sizeof(float));

    for (int r = 0; r < k; r++) {
        double sig = S[r];
        for (int i = 0; i < t; i++) {
            double u = U[i][r];
            size_t base = (size_t)i * (size_t)m;
            for (int j = 0; j < m; j++)
                out[base + j] += (float)(sig * u * V[j][r]);
        }
    }

    return out;
}

int main(void) {
    char infile[256], outfile[256];
    int k;

    printf("Enter input PGM filename:\n");
    if (scanf("%255s", infile) != 1) {
        printf("Invalid input filename.\n");
        return 1;
    }

    printf("Enter output PGM filename:\n");
    if (scanf("%255s", outfile) != 1) {
        printf("Invalid output filename.\n");
        return 1;
    }

    printf("Enter value of k (number of singular values to keep):\n");
    if (scanf("%d", &k) != 1 || k <= 0) {
        printf("Invalid k value.\n");
        return 1;
    }

    int w = 0, h = 0;
    float *A_f = read_pgm(infile, &w, &h);
    if (!A_f) {
        printf("Failed to read %s\n", infile);
        return 1;
    }

    int t = h, m = w;
    double **A = mat(t, m); // A (t x m )
    if (!A) {
        printf("Matrix allocation failed.\n");
        free(A_f);
        return 1;
    }

    for (int i = 0; i < t; i++)
        for (int j = 0; j < m; j++)
            A[i][j] = (double)A_f[i * m + j];

    free(A_f);

    int p = P_DEF, q = Q_DEF;
    int maxnm = (t < m) ? t : m;
    if (k > maxnm) k = maxnm;
    if (k + p > maxnm) p = maxnm - k;

    printf("rSVD: t=%d m=%d k=%d p=%d q=%d\n", t, m, k, p, q);

    srand((unsigned)time(NULL));

    double **U = NULL, **V = NULL, *S = NULL;
    randomized_svd(t, m, A, k, p, q, &U, &S, &V);

    float *out = reconstruct_from_usv(t, m, U, S, V, k);

    {
        double sumA = 0.0, sumDiff = 0.0;
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < m; j++) {
                double aij = A[i][j];
                double ak = out[(size_t)i * m + j];
                sumA += aij * aij;
                double d = aij - ak;
                sumDiff += d * d;
            }
        }
        double normA = sqrt(sumA);
        double normDiff = sqrt(sumDiff);
        printf("Frobenius norm  ||A||_F = %.6f\n", normA);
        printf("Frobenius norm  ||A - A_k||_F = %.6f\n", normDiff);
        printf("Frobenius norm error ||A - A_k||_F/||A||_F = %.6f\n", normDiff/normA);
    }

    for (size_t i = 0; i < (size_t)t * (size_t)m; i++) {
        if (out[i] < 0) out[i] = 0;
        if (out[i] > 1) out[i] = 1;
    }

    if (write_pgm(outfile, out, m, t) != 0)
        printf("Write failed\n");
    else
        printf("Wrote %s (k=%d p=%d q=%d)\n", outfile, k, p, q);

    free(out);

    for (int i = 0; i < t; i++) free(U[i]);
    free(U);

    free(S);

    for (int i = 0; i < m; i++) free(V[i]);
    free(V);

    free_mat(t, A);

    return 0;
}

