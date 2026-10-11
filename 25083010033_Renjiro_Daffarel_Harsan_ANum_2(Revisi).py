# -*- coding: utf-8 -*-

# %% 0. Setup
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PDF_PATH = os.path.join(BASE_DIR, "perahu.pdf")
OUT_DIR = os.path.join(BASE_DIR, "hasil")
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 200              # resolusi render PDF
PAGE_INDEX = 0         # halaman gambar perahu (halaman pertama)
CELL_CM = 45.0         # 1 kotak = 5 garis + 4 kosong = 9 x 5 cm = 45 cm
X_A, X_B = 2.0, 18.0   # batas hitung (kotak): garis reserve buoyancy buritan & haluan
STEP = 0.125           # jarak antar titik ukur (kotak)
cm = CELL_CM


def simpson(y, dx):
    """Aturan Simpson 1/3 komposit untuk titik berjarak seragam (pengganti scipy.integrate.simpson)."""
    y = np.asarray(y, float)
    n = len(y) - 1                      # jumlah pias
    if n < 1:
        return 0.0
    if n % 2 == 0:
        return dx / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())
    # jumlah pias ganjil: Simpson untuk n-1 pias pertama + trapesium untuk pias terakhir
    return simpson(y[:-1], dx) + dx / 2 * (y[-2] + y[-1])


def cubic_spline_integral(x, y):
    """Integral natural cubic spline pada titik berjarak seragam (pengganti scipy CubicSpline)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    h = x[1] - x[0]
    n = len(x)
    M = np.zeros(n)                     # turunan kedua di tiap titik (natural: M0 = Mn = 0)
    if n > 2:
        T = (np.diag(np.full(n - 2, 2 * h / 3)) + np.diag(np.full(n - 3, h / 6), 1) + np.diag(np.full(n - 3, h / 6), -1))
        rhs = (y[2:] - 2 * y[1:-1] + y[:-2]) / h
        M[1:-1] = np.linalg.solve(T, rhs)
    return float(np.sum(h * (y[:-1] + y[1:]) / 2 - h ** 3 / 24 * (M[:-1] + M[1:])))


def save_show(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


def render_pdf(path, dpi, page_index):
    """Render halaman PDF menjadi array RGB. Mencoba PyMuPDF, lalu pdf2image, lalu pypdfium2."""
    errors = []
    try:
        import fitz  # PyMuPDF (tidak butuh poppler, paling mudah di Windows)
        doc = fitz.open(path)
        pix = doc[page_index].get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    except Exception as e:
        errors.append(f"PyMuPDF: {e}")
    try:
        from pdf2image import convert_from_path  # butuh poppler terpasang
        return np.array(convert_from_path(path, dpi=dpi)[page_index].convert("RGB"))
    except Exception as e:
        errors.append(f"pdf2image: {e}")
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        return np.array(pdf[page_index].render(scale=dpi / 72).to_pil().convert("RGB"))
    except Exception as e:
        errors.append(f"pypdfium2: {e}")
    sys.exit("Gagal membaca PDF. Jalankan: pip install pymupdf\n" + "\n".join(errors))


# %% 1. Muat gambar & kalibrasi grid
if not os.path.exists(PDF_PATH):
    sys.exit(f"File tidak ditemukan: {PDF_PATH}\nTaruh perahu.pdf di folder yang sama dengan skrip ini.")

page = render_pdf(PDF_PATH, DPI, PAGE_INDEX).astype(int)
R, G, B = page[..., 0], page[..., 1], page[..., 2]
green = (G - R > 40) & (G - B > 40)                                                  # garis grid hijau
dark = ((R + G + B) / 3 < 200) & ~green & (abs(R - G) < 35) & (abs(G - B) < 35)      # garis lambung


def peaks(a, th):
    idx = np.where(a > th)[0]
    out = []
    s = p = idx[0]
    for i in idx[1:]:
        if i - p > 3:
            out.append((s + p) / 2)
            s = i
        p = i
    out.append((s + p) / 2)
    return np.array(out)


gx = peaks(green.sum(0), green.sum(0).max() * 0.4)   # garis grid vertikal (px)
S = np.median(np.diff(gx))                           # 1 kotak dalam piksel
X0 = gx[0]                                           # garis grid paling kiri = x 0
Y0 = peaks(green.sum(1), green.sum(1).max() * 0.4)[0]
print(f"Ukuran gambar                    : {page.shape[1]} x {page.shape[0]} px")
print(f"Jumlah garis grid vertikal       : {len(gx)}")
print(f"Ukuran 1 kotak di gambar         : {S:.2f} piksel (= {CELL_CM:.0f} cm)")
print(f"Skala                            : 1 piksel = {CELL_CM / S:.3f} cm")

# %% 2. Digitalisasi lebar w(x) dan kedalaman d(x)


def first_dark(px, y0, y1):
    ys = np.where(dark[y0:y1, px])[0]
    return ys[0] + y0 if len(ys) else np.nan


def last_dark(px, y0, y1):
    ys = np.where(dark[y0:y1, px])[0]
    return ys[-1] + y0 if len(ys) else np.nan


col = lambda xc: int(round(X0 + xc * S))

c8 = col(8)
CY1 = (first_dark(c8, 125, 330) + last_dark(c8, 125, 330)) / 2      # sumbu lambung 1
CY2 = (first_dark(c8, 640, 840) + last_dark(c8, 640, 840)) / 2      # sumbu lambung 2
DECK = np.median([first_dark(col(x), 950, 1200) for x in np.arange(16, 17.5, 0.25)])

xs = np.arange(X_A, X_B + 1e-9, STEP)
w1 = np.zeros_like(xs)
w2 = np.zeros_like(xs)
d = np.zeros_like(xs)
beam1 = np.zeros(len(xs), bool)
beam2 = np.zeros(len(xs), bool)
for i, xc in enumerate(xs):
    px = col(xc)
    w1[i] = 2 * (CY1 - first_dark(px, 125, int(CY1))) / S
    w2[i] = 2 * (last_dark(px, int(CY2), 840) - CY2) / S
    d[i] = (last_dark(px, 1100, 1175) - DECK) / S
    beam1[i] = dark[124:131, px - 6:px + 7].any()
    beam2[i] = dark[810:817, px - 6:px + 7].any()

bad = np.abs(w1 - w2) > 0.12
bad1, bad2 = beam1 | bad, beam2 | bad
w1c = np.where(bad1 & ~bad2, w2, w1)
w2c = np.where(bad2 & ~bad1, w1, w2)
both = bad1 & bad2
if both.any():
    w1c[both] = np.interp(xs[both], xs[~both], w1[~both])
    w2c[both] = np.interp(xs[both], xs[~both], w2[~both])
w1, w2 = w1c, w2c

print(f"Pusat lambung 1 : y = {CY1:.1f} px | lambung 2 : y = {CY2:.1f} px | geladak : y = {DECK:.1f} px")
print(f"Jarak antar pusat lambung        : {(CY2 - CY1) / S:.2f} kotak = {(CY2 - CY1) / S * cm / 100:.2f} m")
print(f"Titik ukur                       : {len(xs)} titik, x = {X_A} s/d {X_B} kotak ({(X_B - X_A) * cm / 100:.1f} m)")
print(f"Beda maks lebar lambung 1 vs 2   : {np.abs(w1 - w2).max():.3f} kotak")
if np.isnan(w1).any() or np.isnan(w2).any() or np.isnan(d).any():
    print("PERINGATAN: ada titik ukur NaN, periksa rentang piksel pada sel 2.")

# %% 3. Bukti visual di atas gambar asli
H, W = page.shape[:2]
cell_extent = lambda y0, y1: [(0 - X0) / S, (W - X0) / S, (y1 - Y0) / S, (y0 - Y0) / S]
yc = lambda py: (py - Y0) / S

fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [3.2, 1.2]})
a1.imshow(page[90:860].astype(np.uint8), extent=cell_extent(90, 860))
for ctr, w in [(CY1, w1), (CY2, w2)]:
    a1.fill_between(xs, yc(ctr) - w / 2, yc(ctr) + w / 2, color="tab:blue", alpha=.35)
    a1.plot(xs, yc(ctr) - w / 2, "b-", lw=1.5)
    a1.plot(xs, yc(ctr) + w / 2, "b-", lw=1.5)
for x in np.arange(X_A, X_B + .1, 1):
    a1.axvline(x, color="red", lw=.7, alpha=.6)
for lo, hi in [(-1, X_A), (X_B, 21)]:
    a1.axvspan(lo, hi, color="gray", alpha=.45)
a1.set_xlim(-0.5, 21)
a1.set_title("TAMPAK ATAS - biru = penampang yang dihitung, garis merah = potongan tiap 1 kotak")
a1.set_ylabel("y (kotak)")

a2.imshow(page[930:1200].astype(np.uint8), extent=cell_extent(930, 1200))
bottom = DECK + d * S
a2.fill_between(xs, yc(DECK), yc(bottom), color="tab:orange", alpha=.4)
a2.plot(xs, yc(bottom), color="darkorange", lw=2)
for x in np.arange(X_A, X_B + .1, 1):
    a2.axvline(x, color="red", lw=.7, alpha=.6)
for lo, hi in [(-1, X_A), (X_B, 21)]:
    a2.axvspan(lo, hi, color="gray", alpha=.45)
a2.set_xlim(-0.5, 21)
a2.set_title("TAMPAK SAMPING - oranye = kedalaman d(x)")
a2.set_xlabel("x (kotak grid, 1 kotak = 45 cm)")
save_show("1_bukti_visual.png")

# %% 4. Profil w(x), d(x), A(x)
A1 = (w1 * cm) * (d * cm) / 1e4     # m^2
A2 = (w2 * cm) * (d * cm) / 1e4
xm = (xs - X_A) * cm / 100

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].plot(xm, w1 * cm, label="lambung 1")
ax[0].plot(xm, w2 * cm, "--", label="lambung 2")
ax[0].set_title("Lebar w(x) [cm]")
ax[0].legend()
ax[1].plot(xm, d * cm, color="tab:orange")
ax[1].set_title("Kedalaman d(x) [cm]")
ax[2].fill_between(xm, A1, alpha=.35, color="tab:green")
ax[2].plot(xm, A1, color="tab:green")
ax[2].set_title("Luas penampang A(x) = w*d [m2]")
for a in ax:
    a.set_xlabel("jarak dari garis reserve buoyancy buritan [m]")
    a.grid(alpha=.3)
save_show("2_profil.png")

# %% 4b. Potongan melintang (persegi panjang)
fig, ax = plt.subplots(1, 5, figsize=(15, 3.2))
for a, xc in zip(ax, [2, 6, 10, 14, 18]):
    i = int(np.argmin(np.abs(xs - xc)))
    wcm, dcm = w1[i] * cm, d[i] * cm
    a.add_patch(Rectangle((-wcm / 2, -dcm), wcm, dcm, fc="lightsteelblue", ec="navy", lw=2))
    a.text(0, -dcm / 2, f"A = {wcm / 100:.2f} x {dcm / 100:.2f}\n= {wcm * dcm / 1e4:.3f} m2", ha="center", va="center")
    a.set_xlim(-130, 130)
    a.set_ylim(-150, 15)
    a.set_aspect("equal")
    a.axhline(0, color="saddlebrown", lw=3)
    a.set_title(f"x = {xc} kotak ({xc * cm / 100:.2f} m)")
    a.set_xlabel(f"w = {wcm:.0f} cm")
plt.suptitle("Tampak depan: potongan melintang = persegi panjang", y=1.02)
save_show("3_potongan_melintang.png")

# %% 5. Tabel perhitungan manual Simpson, h = 1 kotak
h = 1.0 * cm / 100
idx = np.arange(0, len(xs), int(round(1.0 / STEP)))
xi, wi, di = xs[idx], w1[idx] * cm / 100, d[idx] * cm / 100
Ai = wi * di
n = len(xi) - 1
assert n % 2 == 0, "Simpson 1/3 butuh jumlah pias genap"
coef = np.array([1] + [4 if k % 2 else 2 for k in range(1, n)] + [1])
print(f"{'i':>2} {'x (m)':>7} {'w (m)':>7} {'d (m)':>7} {'A (m2)':>9} {'koef':>5} {'koef*A':>9}")
for k in range(n + 1):
    print(f"{k:>2} {(xi[k] - X_A) * cm / 100:>7.2f} {wi[k]:>7.3f} {di[k]:>7.3f} {Ai[k]:>9.4f} {coef[k]:>5d} {coef[k] * Ai[k]:>9.4f}")
V_trap1 = h / 2 * (Ai[0] + 2 * Ai[1:-1].sum() + Ai[-1])
V_simp1 = h / 3 * (coef * Ai).sum()
print(f"\nSigma(koef*A)            = {(coef * Ai).sum():.4f}")
print(f"V Simpson (h/3)*Sigma    = ({h:.2f}/3)*{(coef * Ai).sum():.4f} = {V_simp1:.4f} m3")
print(f"V Trapesium (h=1 kotak)  = {V_trap1:.4f} m3")

# %% 6. Uji konvergensi


def V_of(Aprof, step_cells, method):
    k = int(round(step_cells / STEP))
    a = Aprof[::k]
    hh = step_cells * cm / 100
    if method == "trap":
        return hh * (a.sum() - 0.5 * (a[0] + a[-1]))
    return simpson(a, dx=hh)


V_ref = V_of(A1, STEP, "simp")
print(f"{'h (kotak)':>10} {'h (cm)':>7} {'pias':>5} {'Trapesium':>11} {'Simpson':>9} {'selisih Simpson-ref':>20}")
hs = [2, 1, 0.5, 0.25, 0.125]
errT, errS = [], []
for hh in hs:
    vt, vs = V_of(A1, hh, "trap"), V_of(A1, hh, "simp")
    errT.append(abs(vt - V_ref))
    errS.append(abs(vs - V_ref))
    print(f"{hh:>10} {hh * cm:>7.1f} {int((X_B - X_A) / hh):>5} {vt:>11.4f} {vs:>9.4f} {vs - V_ref:>+20.5f}")
spl = cubic_spline_integral(xm, A1)
print(f"\nPembanding Cubic Spline natural (semua titik) = {spl:.4f} m3")

plt.figure(figsize=(5.5, 4))
plt.loglog(np.array(hs[:-1]) * cm, np.maximum(errT[:-1], 1e-7), "o-", label="Trapesium (~ h^2)")
plt.loglog(np.array(hs[:-1]) * cm, np.maximum(errS[:-1], 1e-7), "s-", label="Simpson (~ h^4)")
plt.xlabel("h [cm]")
plt.ylabel("|selisih terhadap h terkecil| [m3]")
plt.grid(alpha=.3, which="both")
plt.legend()
plt.title("Konvergensi terhadap ukuran pias h")
save_show("4_konvergensi.png")

# %% 7. Hasil akhir
V1 = V_of(A1, STEP, "simp")
V2 = V_of(A2, STEP, "simp")
Vt = V1 + V2
rho = 1025.0
L = (X_B - X_A) * cm / 100
print("=" * 58)
print(f" Volume lambung 1 (Simpson) : {V1:8.3f} m3 = {V1 * 1000:9.0f} liter")
print(f" Volume lambung 2 (Simpson) : {V2:8.3f} m3 = {V2 * 1000:9.0f} liter")
print(f" VOLUME TOTAL KEDUA LAMBUNG : {Vt:8.3f} m3 = {Vt * 1000:9.0f} liter")
print("=" * 58)
print(f"Panjang yang dihitung        : {L:.2f} m")
print(f"Lebar rata-rata              : {np.mean(w1) * cm / 100:.2f} m | Kedalaman rata-rata: {np.mean(d) * cm / 100:.2f} m")
print(f"Koefisien prismatik          : {V1 / (L * np.max(w1) * cm / 100 * np.max(d) * cm / 100):.3f}")
print(f"Gaya apung maks (air laut)   : {rho * Vt / 1000:.1f} ton = {rho * 9.81 * Vt / 1000:.0f} kN")
print("\nSensitivitas terhadap skala (V ~ skala^3):")
for s in [40, 45, 50]:
    print(f"   1 kotak = {s} cm -> total {Vt * (s / CELL_CM) ** 3:7.3f} m3")

# %% 8. Model 3D
Xm, Wm, Dm = xm, w1 * cm / 100, d * cm / 100
sep = (CY2 - CY1) / S * cm / 100


def hull_faces(y0):
    f = []
    for i in range(len(Xm) - 1):
        x0, x1 = Xm[i], Xm[i + 1]
        f.append([(x0, y0 - Wm[i] / 2, 0), (x1, y0 - Wm[i + 1] / 2, 0), (x1, y0 - Wm[i + 1] / 2, -Dm[i + 1]), (x0, y0 - Wm[i] / 2, -Dm[i])])
        f.append([(x0, y0 + Wm[i] / 2, 0), (x1, y0 + Wm[i + 1] / 2, 0), (x1, y0 + Wm[i + 1] / 2, -Dm[i + 1]), (x0, y0 + Wm[i] / 2, -Dm[i])])
        f.append([(x0, y0 - Wm[i] / 2, -Dm[i]), (x1, y0 - Wm[i + 1] / 2, -Dm[i + 1]), (x1, y0 + Wm[i + 1] / 2, -Dm[i + 1]), (x0, y0 + Wm[i] / 2, -Dm[i])])
        f.append([(x0, y0 - Wm[i] / 2, 0), (x1, y0 - Wm[i + 1] / 2, 0), (x1, y0 + Wm[i + 1] / 2, 0), (x0, y0 + Wm[i] / 2, 0)])
    for j in (0, -1):
        f.append([(Xm[j], y0 - Wm[j] / 2, 0), (Xm[j], y0 + Wm[j] / 2, 0), (Xm[j], y0 + Wm[j] / 2, -Dm[j]), (Xm[j], y0 - Wm[j] / 2, -Dm[j])])
    return f


fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(111, projection="3d")
for y0, c in [(-sep / 2, "steelblue"), (sep / 2, "seagreen")]:
    ax.add_collection3d(Poly3DCollection(hull_faces(y0), facecolor=c, edgecolor="none", alpha=.9))
ax.set_xlim(Xm[0], Xm[-1])
ax.set_ylim(-sep / 2 - 1, sep / 2 + 1)
ax.set_zlim(-1.5, 0.1)
ax.set_box_aspect((Xm[-1] - Xm[0], sep + 2, 1.6))
ax.view_init(24, -58)
ax.set_xlabel("panjang [m]")
ax.set_ylabel("lebar [m]")
ax.set_zlabel("tinggi [m]")
ax.set_title(f"Model 3D kedua lambung (tanpa reserve buoyancy) - V total = {Vt:.2f} m3")
save_show("5_model_3d.png")

print(f"\nSemua gambar tersimpan di: {OUT_DIR}")
