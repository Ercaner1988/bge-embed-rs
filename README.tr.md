# bge-embed-rs

[English](README.md)

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) için tek dosyalık, saf Rust (candle) tabanlı bir embedding sunucusu. OpenAI uyumlu bir `/v1/embeddings` API'si ve küçük bir masaüstü penceresi sunar. Tamamen yerelde, CPU üzerinde çalışır. Python yok, Docker yok, harici bir çalışma zamanı yok.

## İndirme

En güncel sürümü [Releases sayfasından](https://github.com/Ercaner1988/bge-embed-rs/releases) alın. `release.yml` her etiket için şu altı dosyayı üretir:

| Dosya | Platform |
|---|---|
| `bge-embed-rs-x86_64-pc-windows-msvc.exe` | Windows, x86-64 |
| `bge-embed-rs-aarch64-pc-windows-msvc.exe` | Windows, ARM64 |
| `bge-embed-rs-x86_64-unknown-linux-gnu` | Linux, x86-64 |
| `bge-embed-rs-aarch64-unknown-linux-gnu` | Linux, ARM64 |
| `bge-embed-rs-x86_64-apple-darwin` | macOS, Intel |
| `bge-embed-rs-aarch64-apple-darwin` | macOS, Apple Silicon |

x86-64 derlemeleri `x86-64-v3` hedefiyle yapılır (AVX2 + FMA + BMI — Intel Haswell 2013 ve sonrası, AMD Excavator/Zen 2015 ve sonrası). Bu özellikleri desteklemeyen eski bir işlemcide uygulama çökmek yerine başlamayı reddeder ve nedenini söyler.

CI, her derlemeyi yayımlamadan önce çalıştırıp bir deneme cümlesini embed eder. Tek istisna macOS Intel derlemesi: Apple Silicon üzerinde çapraz derleniyor ve Rosetta'da AVX2 olmadığı için CI'da derlenir ama çalıştırılmaz.

## İlk çalıştırma

Dosyaya çift tıklayın. Bir pencere açılır ve modeli (BAAI/bge-m3, ~2,2 GB) Hugging Face hub'ından, standart HF önbelleğine, ilerleme çubuğuyla birlikte bir kerelik indirir. Sonraki çalıştırmalar önbellekteki dosyaları kullanır, tekrar indirme yapmaz.

Sunucularda pencereye gerek yoksa `--headless` parametresi veya `BGE_HEADLESS=1` ortam değişkeni pencereyi atlayıp sıradan bir CLI sunucusu gibi çalıştırır.

## İmzasız ikili dosyalar

Bu derlemeler kod imzalı değildir, bu yüzden işletim sisteminiz ilk çalıştırmada uyarır:

- **Windows**: SmartScreen — "Diğer bilgiler" sonra "Yine de çalıştır"a tıklayın.
- **macOS**: Gatekeeper — dosyaya sağ tıklayıp "Aç"ı seçin, ya da `xattr -d com.apple.quarantine <dosya>` komutunu çalıştırın.
- **Linux**: önce çalıştırılabilir yapın: `chmod +x <dosya>`.

## API

`POST /v1/embeddings` — gövde `{"model": "bge-m3", "input": "metin"}` ya da `{"model": "bge-m3", "input": ["metin1", "metin2"]}` şeklinde. `model` alanı kabul edilir ama yok sayılır (bu sunucunun çalıştırdığı tek model bge-m3'tür). 1024 boyutlu, L2-normalize edilmiş, CLS havuzlamalı vektörler döner.

`GET /health` — sunucu ayaktaysa 200 OK döner.

```bash
curl http://127.0.0.1:11435/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":"merhaba dünya"}'
```

## Yapılandırma

Başlangıçta okunan ortam değişkenleri:

| Değişken | Varsayılan | Not |
|---|---|---|
| `BGE_HOST` | `127.0.0.1` | `0.0.0.0` sunucuyu LAN'a/Docker'a açar — kimlik doğrulama yoktur, yalnızca güvendiğiniz bir ağda kullanın. |
| `BGE_PORT` | `11435` | |
| `BGE_PARALLEL` | `4` | Embedding işini yürüten iş parçacığı sayısı. |
| `BGE_HEADLESS` | tanımsız | `1` pencereyi devre dışı bırakır. |

## Araçlara bağlanma

Connections ekranı yerel RAG araçlarını algılar ve bağlar:

- **Open Notebook, Open WebUI, AnythingLLM** — HTTP üzerinden taranır, bulunduğunda otomatik bağlanır.
- **LibreChat, Dify** — HTTP taraması yapılmaz; ekran bunun yerine yapıştırmaya hazır bir yapılandırma parçacığı için "Copy config" düğmesi sunar.

"Runs in Docker" anahtarı, araca verilen adresi `127.0.0.1`'den `host.docker.internal`'a çevirir; aracın kendisi bir konteynerde çalışıyorsa kullanılır.

Bağlanma, mevcut modellerinizi ya da kimlik bilgilerinizi hiçbir zaman silmez veya değiştirmez; araç zaten bu sunucuyu kullanıyorsa hiçbir şey yapmaz. Yine de bir aracın embedding modelini değiştirmenin sonuçları vardır:

- **Open Notebook**: tutarlı arama için mevcut kaynakların yeniden embed edilmesi gerekebilir. Uygulama varsayılan modeli değiştirdikten sonra bir not gösterir.
- **Open WebUI**: bilgi tabanlarının yeniden indekslenmesi gerekir. Uygulama değişiklikten sonra bir not gösterir.
- **AnythingLLM**: embedder'ı değiştirmek **her çalışma alanının embed edilmiş tüm belgelerini siler** (AnythingLLM vektör veritabanını sıfırlar). Uygulama, "Delete and connect" ile onaylamadan hiçbir şey yazmaz.

## Performans

Ryzen 5 7430U dizüstü bilgisayarda (6 çekirdek / 12 iş parçacığı), ~1.500 karakterlik parçalarla, `BGE_PARALLEL=4` ayarında ölçüldü: istek başına parça başına yaklaşık 1,9–2,5 saniye.

Taşınabilir `x86-64-v3` sürümü, aynı makinedeki bir `target-cpu=native` derlemesiyle aynı performansı veriyor: medyan 2,49 sn/parça'ya karşı 2,78 sn/parça, aynı vektörler (cosine 1.000000).

Vektörler, bir llama.cpp bge-m3 referans sunucusuyla ≥ 0,9999 cosine benzerliğinde eşleşiyor (CLS havuzlama, L2 normalizasyon).

Tekrarlamak için farklı portlarda iki sunucu çalıştırın ve [Bun](https://bun.sh) kurulu olsun:

```bash
BENCH_SOURCE=book.txt bun bench.ts native=11434 v3=11435
```

### Eşzamanlılık: kısa sorgular ve toplu işler

Tek bir istek zaten `BGE_PARALLEL` kadar iş parçacığını meşgul ediyor. Birden fazla istemci
sunucuyu aynı anda çağırdığında (ör. bir sohbet aracı toplu yeniden indeksleme yaparken başka bir
araç canlı bir arama sorgusu gönderiyor), her eşzamanlı çok-girdili istek kendi `BGE_PARALLEL`
iş parçacığını ekliyor ve kısa, tek-girdili bir sorgu aç kalıyor: ölçülen medyan gecikme boşta
~0,65 sn'den, bir eşzamanlı toplu iş altında 7,4 sn'ye, üç eşzamanlı toplu iş altında 47 sn'ye çıktı.

`BGE_TOPLU_IZIN` (varsayılan 1) çok-girdili istekleri bir semafordan geçiriyor; ~512 karakterin
altındaki tek-girdili bir istek (tipik bir arama sorgusu) kapıyı her zaman atlıyor. Bu, kısa-sorgu
gecikmesi karşılığında bir miktar toplu verimden ödün veriyor: izin 1 medyanı ~1,8 sn'ye
düşürdü, bedeli 3 eşzamanlı istemciyle 3 dönen turda ölçülen yaklaşık %27 toplu verim kaybı.

Tekrarlamak için:

```bash
BENCH_SOURCE=book.txt bun lane-bench.ts gated=./target/release/bge-embed-rs.exe|1 ungated=./target/release/bge-embed-rs.exe|999
```

## Kaynaktan derleme

```bash
cargo build --release
```

`.cargo/config.toml`, `x86_64` derlemeleri için `target-cpu=x86-64-v3`'ü otomatik ayarlar: candle'ın eleman bazlı işlemleri yalnız derleme hedefinin izin verdiği SIMD komutlarını kullanır ve SSE2 tabanlı bir derleme yaklaşık 10 kat yavaştı.

## Teşekkürler ve lisanslar

Kod [MIT lisanslıdır](LICENSE).

Sunucu [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) modelini indirip çalıştırır; kullanmadan önce lisansını o sayfadan kontrol edin.

Pakete dahil Inter yazı tipi [SIL Open Font License 1.1](assets/fonts/LICENSE.txt) ile lisanslıdır.

[candle](https://github.com/huggingface/candle), [egui/eframe](https://github.com/emilk/egui), [axum](https://github.com/tokio-rs/axum) ve [hf-hub](https://github.com/huggingface/hf-hub) ile geliştirildi.
