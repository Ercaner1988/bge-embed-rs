# bge-dizin

bge-m3 gömmeleri için saf Rust HNSW yan dizini. Sunucunun (İbnü'n-Nedîm Gömme, `ibnun-nedim`) ürettiği vektörleri hızlı yakın-komşu aramasına açar. Doğruluk kaynağı çağıranın veritabanıdır (el-Fihrist'te Turso tabloları); dizin her an yeniden kurulabilir.

**Neden:** Turso 0.8.2'de yoğun vektör dizini yok (el-Fihrist ADR 0005). 100 bin × 1024 vektörde `vector_distance_cos` tam taraması, yol haritasının 50 ms eşiğinin çok üstünde ölçüldü.

## Kullanım

```rust
use bge_dizin::{Ayar, Dizin};
let mut d = Dizin::yeni(1024, Ayar::default())?;
d.ekle(satir_kimligi, &vektor)?;          // L2 normalize edilir; canlı kimlik tekrarı ret
let sonuc = d.ara(&sorgu, 10, 128)?;       // (kimlik, kosinüs) azalan; ef = 128
d.sil(satir_kimligi);                      // mezar taşı; sonra aynı kimlik yeniden eklenebilir
d.kaydet(Path::new("dizin.bged"))?;        // rkyv, başlıklı, atomik
let d = Dizin::yukle(Path::new("dizin.bged"))?;
```

- Dönen benzerlikler **kesindir** (dizin tam vektörleri tutar); yaklaşık olan yalnız aday kümesidir. Bu yüzden veritabanında ayrıca yeniden sıralamaya gerek yoktur.
- `ef` büyüdükçe isabet artar, hız düşer.

## Tasarım

- **HNSW** (Malkov ve Yashunin, 2018), komşu seçiminde çeşitlilik sezgisi. `m = 16`, `ef_insa = 100`, seviyeler tohumdan belirlenimli.
- **Kalıcılık:** rkyv (altin-kapi ADR 0005). Dosyada 16 baytlık başlık var (`BGED`, sürüm, sıfırlar). Yük hizalı tampondan doğrulanır, ardından anlamsal tutarlılık denetlenir (sınır dışı komşu, giriş noktası, canlı eşlem). Bozuk, yabancı ya da sürümü tutmayan dosya açık hata verir; sessiz boş sonuç yoktur.
- Kurulum tek iş parçacığında. Paralel kurulum ölçüm gerektirirse eklenir.

## Ölçüm

```sh
# makine ölçümü (sentetik, kümelenmiş; gerçek gömmeleri temsil etmez)
cargo run --release -p bge-dizin --example olc -- sentetik 100000 1024 200
# gerçek veri: önce metinleri ibnun-nedim ile göm, sonra ölç
bun crates/bge-dizin/deneme/gom.ts --klasor <metin klasörü> --cikti vektorler.f32
cargo run --release -p bge-dizin --example olc -- gercek vektorler.f32 1024
```

Kabul eşiği (el-Fihrist yol haritası, Görev 2): recall@10 ≥ 0,95 ve p50 < 20 ms. Sorgular dizine konmayan ayrık bir örnektir; isabet, bellekte kaba kuvvete göre hesaplanır.

Ölçüm (2026-10-06): **sentetik** kümelenmiş veri (200 küme), 99 800 × 1024 vektör, 200 ayrık sorgu. 4 çekirdekli Linux bulut konteyneri, release derleme; Windows makinesini temsil etmez.

| ef | recall@10 | p50 | p95 |
|---|---|---|---|
| 32 | 0,9520 | 0,16 ms | 0,50 ms |
| 64 | 0,9695 | 0,18 ms | 0,50 ms |
| 128 | 0,9795 | 0,23 ms | 0,53 ms |
| 256 | 0,9845 | 0,32 ms | 0,62 ms |

- Kurulum 69,9 s (tek iş parçacığı).
- Bellekte kaba kuvvet p50 123,8 ms.
- Dosya 403,9 MiB; yazma 5,83 s, yükleme 2,52 s.

Dört ef değerinin hepsi kabul eşiğini geçiyor. Gerçek veriyle ölçülmedi.

**Bilinen sınır:** dağılım dışı sorgularda (rastgele yönlerde, verinin kümelerinden uzak) isabet düşer. 3000 × 64'lük bir sınamada ef = 64'te 0,93 ölçüldü. Gerçek kullanımda sorgu da aynı modelin gömmesi olduğu için bu durum beklenmez; yine de gerçek veriyle ölçülmeli.
