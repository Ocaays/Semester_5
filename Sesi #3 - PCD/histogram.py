
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

# 1. Membaca gambar RGB
gambar = iio.imread("gambar.JPG")

# 2. Mengubah RGB menjadi grayscale
if gambar.ndim == 3:
    if gambar.shape[2] == 4:
        gambar = gambar[:, :, :3]

    grayscale = np.dot(
        gambar[:, :, :3],
        [0.299, 0.587, 0.114]
    ).astype(np.uint8)
else:
    grayscale = gambar.astype(np.uint8)

# 3. Menghitung histogram intensitas 0-255
histogram = np.bincount(
    grayscale.ravel(),
    minlength=256
)

# 4. Menampilkan jumlah piksel
print("Ukuran gambar:", grayscale.shape)
print("Total piksel:", grayscale.size)

for intensitas in range(256):
    print(f"Intensitas {intensitas}: {histogram[intensitas]} piksel")

# 5. Menentukan intensitas dominan
dominan = np.argmax(histogram)
print("Intensitas dominan:", dominan)
print("Jumlah piksel dominan:", histogram[dominan])

# 6. Menampilkan gambar grayscale dan histogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(grayscale, cmap="gray", vmin=0, vmax=255)
plt.title("Gambar Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.plot(range(256), histogram, color="black")
plt.title("Histogram Grayscale")
plt.xlabel("Intensitas Piksel (0-255)")
plt.ylabel("Jumlah Piksel")
plt.xlim(0, 255)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hasil_histogram.png")
plt.show()
