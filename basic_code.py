import numpy as np
import cv2
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

def grayscale_intensity(image):
    """Görüntüyü gri tonlamaya çevirir ve yoğunluk haritası oluşturur."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Yoğunluk haritası - koyu yerler daha yüksek yoğunluklu
    intensity = 255 - gray
    return intensity

def generate_weighted_points(intensity, n_points=1000):
    """Yoğunluk haritasına göre ağırlıklı rastgele noktalar oluşturur."""
    # Yoğunluk haritasını normalize et
    if intensity.max() > 0:
        normalized = intensity / intensity.max()
    else:
        normalized = intensity
    
    # Daha verimli nokta seçimi - sadece sıfır olmayan piksellere odaklan
    y_indices, x_indices = np.nonzero(normalized)
    probabilities = normalized[y_indices, x_indices]
    probabilities = probabilities / probabilities.sum()
    
    # Rastgele nokta indeksi seçimi (ağırlıklı)
    selected_indices = np.random.choice(
        len(y_indices), 
        size=min(n_points, len(y_indices)), 
        p=probabilities, 
        replace=False
    )
    
    # Seçilen noktaları al
    points = np.column_stack([x_indices[selected_indices], y_indices[selected_indices]])
    return points

def lloyd_relaxation(points, intensity, iterations=10):
    """Lloyd gevşetme algoritması ile noktaları dağıt."""
    height, width = intensity.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    coords = np.vstack((x_coords.ravel(), y_coords.ravel())).T
    
    for _ in range(iterations):
        # Voronoi diyagramı oluştur
        vor = Voronoi(points)
        
        # Her bölge için ağırlık merkezi hesapla
        new_points = []
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                new_points.append(points[i])
                continue
                
            # Bölge poligonunu al
            polygon = [vor.vertices[v] for v in region]
            if not polygon:
                new_points.append(points[i])
                continue
                
            # Poligonu sınırlandır
            polygon_array = np.array(polygon)
            x_min, y_min = np.floor(polygon_array.min(axis=0)).astype(int)
            x_max, y_max = np.ceil(polygon_array.max(axis=0)).astype(int)
            
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width-1, x_max)
            y_max = min(height-1, y_max)
            
            if x_min >= x_max or y_min >= y_max:
                new_points.append(points[i])
                continue
            
            # Poligon içindeki pikselleri kontrol et
            x_range = np.arange(x_min, x_max+1)
            y_range = np.arange(y_min, y_max+1)
            xx, yy = np.meshgrid(x_range, y_range)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T
            
            # Daha hızlı centroid hesaplama
            weights = np.array([intensity[y, x] if 0 <= y < height and 0 <= x < width else 0 
                              for x, y in pixels])
            
            if weights.sum() > 0:
                centroid = np.average(pixels, axis=0, weights=weights)
                new_points.append(centroid)
            else:
                new_points.append(points[i])
        
        points = np.array(new_points)
    
    return points

def detect_object_with_stippling(image_path, n_points=2000, lloyd_iterations=5):
    """Weighted Voronoi Stippling ile nesne tespiti."""
    # Görüntüyü yükle
    start_time = time.time()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    
    print(f"Görüntü boyutu: {image.shape}")
    
    # Gri tonlama ve yoğunluk haritası oluştur
    intensity = grayscale_intensity(image)
    
    # Ağırlıklı rastgele noktalar oluştur
    points = generate_weighted_points(intensity, n_points)
    
    print(f"Başlangıç noktaları oluşturuldu: {len(points)} nokta")
    
    # Lloyd algoritması ile nokta dağılımını iyileştir
    relaxed_points = lloyd_relaxation(points, intensity, lloyd_iterations)
    
    print(f"Lloyd gevşetme tamamlandı, {len(relaxed_points)} nokta")
    
    # Sonuçları görselleştir
    plt.figure(figsize=(12, 8))
    
    # Orijinal görüntü
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal Görüntü")
    plt.axis('off')
    
    # Stippling sonucu
    plt.subplot(1, 2, 2)
    plt.imshow(np.ones_like(image) * 255, cmap='gray')
    plt.scatter(relaxed_points[:, 0], relaxed_points[:, 1], s=1, c='black')
    plt.title("Weighted Voronoi Stippling")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Sonucu kaydet
    output_path = image_path.rsplit('.', 1)[0] + "_stippled.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"İşlem tamamlandı. Süre: {time.time() - start_time:.2f} saniye")
    print(f"Sonuç kaydedildi: {output_path}")
    
    plt.show()
    return relaxed_points

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Örnek bir resim yolu belirtin
        image_path = input("Lütfen bir görüntü dosyası yolu girin: ")
    
    # Parametre ayarları
    n_points = 3000  # Nokta sayısı
    lloyd_iterations = 3  # Lloyd gevşetme iterasyonları
    
    # Nesne tespiti yap
    stipple_points = detect_object_with_stippling(
        image_path, 
        n_points=n_points, 
        lloyd_iterations=lloyd_iterations
    )