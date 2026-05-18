# Bitki Hastalıkları Tespiti için Evrişimli Sinir Ağı (CNN)

Bu proje, bitkilerdeki hastalıkları tespit etmek amacıyla Evrişimli Sinir Ağları (CNN) kullanarak bir görüntü sınıflandırma modeli geliştirmeyi amaçlamaktadır. Proje, görüntü işleme ve derin öğrenme tekniklerini kullanarak bitki hastalıklarının erken teşhisine katkıda bulunmayı hedeflemektedir.

## İçindekiler

- [Proje Amacı](#proje-amacı)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Veri Seti](#veri-seti)
- [Model Mimarisi](#model-mimarisi)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Eğitim ve Değerlendirme](#eğitim-ve-değerlendirme)
- [Sonuçlar](#sonuçlar)
- [Görselleştirmeler](#görselleştirmeler)

## Proje Amacı

Bu projenin temel amacı, bitki yaprak görüntülerinden hastalıkları otomatik olarak tespit edebilen ve sınıflandırabilen sağlam bir derin öğrenme modeli oluşturmaktır. Bu, özellikle tarımda verimliliği artırmak ve ürün kayıplarını en aza indirmek için önemlidir.

## Kullanılan Teknolojiler

- **Python**
- **TensorFlow / Keras**: Derin öğrenme modeli oluşturma ve eğitme için.
- **NumPy**: Sayısal işlemler ve veri manipülasyonu için.
- **Matplotlib**: Veri ve model performansını görselleştirmek için.
- **ImageDataGenerator**: Görüntü veri büyütme (data augmentation) ve ön işleme için.

## Veri Seti

Proje, [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) veri setini kullanmaktadır. Bu veri seti, farklı bitki türlerine ait sağlıklı ve hastalıklı yaprak görüntülerinden oluşmaktadır. Veri seti, Google Drive üzerinden `archive.zip` olarak yüklenip `/content/dataset/PlantVillage` dizinine çıkarılmaktadır.

## Model Mimarisi

Kullanılan model, peş peşe üç adet Evrişimli Katman (Conv2D) ve Maksimum Havuzlama (MaxPooling2D) katmanından oluşan bir CNN mimarisidir. Evrişimli katmanlar, resimdeki özellikleri (kenarlar, dokular, lezyonlar) çıkarmak için kullanılırken, havuzlama katmanları boyut indirgemesi yapar. Çıkarılan özellikler daha sonra `Flatten` katmanı aracılığıyla tam bağlantılı (Dense) katmanlara aktarılır. Modelde aşırı öğrenmeyi (overfitting) engellemek için `Dropout` katmanı da kullanılmıştır. Çıkış katmanı, bitki hastalıklarının sınıf sayısına göre `softmax` aktivasyon fonksiyonu ile olasılık dağılımı sağlar.

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

## Kurulum ve Çalıştırma

1.  **Google Colab Ortamını Hazırlama:**
    *   Bu notebook'u Google Colab'da açın.
    *   GPU hızlandırıcısını etkinleştirin (`Runtime` -> `Change runtime type` -> `GPU`).

2.  **Google Drive Bağlantısı:**
    *   Notebook'un başında Google Drive'ınızı bağlayın. Bu, veri setinize erişim sağlar.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3.  **Veri Setini Yükleme ve Çıkarma:**
    *   `archive.zip` dosyanızın Google Drive'daki doğru yolunu `DATASET_PATH` değişkeninde belirtin.
    *   Zip dosyasını `/content/dataset` dizinine çıkarın.

4.  **Veri Ön İşleme:**
    *   `ImageDataGenerator` kullanarak verileri büyütün (rotation, zoom, horizontal flip) ve pikselleri normalize edin.
    *   `train_generator` ve `validation_generator` oluşturulur.

5.  **Modelin Tanımlanması ve Derlenmesi:**
    *   Yukarıda belirtilen CNN mimarisini oluşturun ve `adam` optimizörü ile `categorical_crossentropy` kaybını kullanarak modeli derleyin.

6.  **Modelin Eğitilmesi:**
    *   `model.fit` fonksiyonunu kullanarak modeli `train_dataset` üzerinde eğitin ve `validation_dataset` üzerinde doğrulayın. Eğitim süreci 10 epoch olarak ayarlanmıştır.

## Eğitim ve Değerlendirme

Model, 10 epoch boyunca eğitim veri seti üzerinde eğitildi ve doğrulama veri seti üzerinde performans değerlendirmesi yapıldı. Eğitim sonrası modelin doğrulama doğruluğu yaklaşık **%89.16** ve doğrulama kaybı yaklaşık **0.3228** olarak gözlemlenmiştir.

## Sonuçlar

Eğitilen model, bitki yapraklarındaki hastalıkları tanıma konusunda başarılı bir performans sergilemiştir. Doğrulama setindeki yüksek doğruluk oranı, modelin yeni, daha önce görmediği görüntüler üzerinde iyi genellenebildiğini göstermektedir.

## Görselleştirmeler

Notebook, eğitim ve doğrulama doğruluğu/kaybı grafiklerini ve modelin rastgele örnekler üzerindeki tahminlerini (gerçek etiket ve tahmin edilen etiketle birlikte) gösteren görselleştirmeler içerir. Bu görseller, modelin öğrenme sürecini ve hata yaptığı durumları anlamanıza yardımcı olur.



