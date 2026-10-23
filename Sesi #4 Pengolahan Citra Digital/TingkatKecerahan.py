
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 1. Membaca gambar berwarna
gambar = iio.imread("gambar.JPG")

# Memastikan gambar menggunakan RGB, bukan RGBA
if gambar.ndim == 3 and gambar.shape[2] == 4:
    gambar = gambar[:, :, :3]

# Pastikan gambar memiliki 3 kanal warna
if gambar.ndim != 3 or gambar.shape[2] != 3:
    raise ValueError("Gunakan gambar berwarna RGB.")

# 2. Mengurangi noise dengan median filter
hasil_filter = np.stack([
    ndimage.median_filter(gambar[:, :, kanal], size=3)
    for kanal in range(3)
], axis=2)

# 3. Meningkatkan kontras setiap kanal warna

# 3. Meningkatkan kecerahan dan kontras secara lebih jelas
data = hasil_filter.astype(np.float32)

# Meningkatkan kecerahan sebesar 35%
lebih_cerah = data * 1.35

# Meningkatkan kontras di sekitar nilai tengah 128
hasil_enhance = (lebih_cerah - 128) * 1.25 + 128

# Membatasi nilai piksel ke rentang 0-255
hasil_enhance = np.clip(
    hasil_enhance, 0, 255
).astype(np.uint8)

# 4. Menampilkan perbandingan
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(gambar)
plt.title("Citra Awal (RGB)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(hasil_filter)
plt.title("Setelah Median Filter")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(hasil_enhance)
plt.title("Citra Lebih Cerah dan Kontras")
plt.axis("off")

plt.tight_layout()
plt.savefig("hasil_enhancement.png")
plt.show()

# 5. Menyimpan citra hasil
iio.imwrite("citra_ditingkatkan.png", hasil_enhance)

print("Peningkatan kecerahan dan kontras berhasil!")

# 4. Menampilkan perbandingan citra
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gambar)
plt.title("Citra Awal (RGB)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(hasil_filter)
plt.title("Setelah Median Filter")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(hasil_enhance)
plt.title("Citra Ditingkatkan")
plt.axis("off")

plt.tight_layout()
plt.savefig("hasil_enhancement.png")
plt.show()

# 5. Menyimpan hasil gambar
iio.imwrite("citra_ditingkatkan.png", hasil_enhance)

print("Peningkatan kualitas citra berhasil!")
print("Ukuran citra:", gambar.shape)
print("Hasil disimpan sebagai citra_ditingkatkan.png")
print("Perbandingan disimpan sebagai hasil_enhancement.png")

