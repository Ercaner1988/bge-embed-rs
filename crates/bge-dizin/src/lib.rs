//! bge-dizin: bge-m3 gömmeleri için saf Rust HNSW yan dizini.
//!
//! **Neden.** Turso 0.8.2'de yoğun vektör dizini yok (el-Fihrist ADR 0005); 100 bin ×
//! 1024 vektörde `vector_distance_cos` tam taraması yol haritasının 50 ms eşiğinin çok
//! üstünde kaldı. Doğruluk kaynağı veritabanı tablolarıdır: bu dizin her an yeniden
//! kurulabilir, kaybolursa veri kaybolmaz.
//!
//! **Yöntem.** HNSW (Malkov ve Yashunin, 2018): katmanlı yakınlık grafiği; komşu
//! seçiminde çeşitlilik sezgisi (makalenin 4. algoritması, budananlarla doldurma).
//! Benzerlik kosinüstür: vektörler eklenirken L2'ye göre normalize edilir, kosinüs iç
//! çarpıma iner. Dizin tam vektörleri tuttuğu için döndürdüğü benzerlikler kesindir;
//! yaklaşık olan yalnız aday kümesidir.
//!
//! **Kalıcılık.** rkyv (altin-kapi ADR 0005): 16 baytlık başlık (4 imza, 4 sürüm LE,
//! 8 sıfır), yük hizalı tampondan doğrulanarak okunur, ardından anlamsal tutarlılık
//! denetlenir. Bozuk, yabancı ya da sürümü tutmayan dosya açık hata verir.
//!
//! **Silme** mezar taşıyla: düğüm gezinti için grafikte kalır, sonuçlara girmez.
//! Aynı kimlik silindikten sonra yeniden eklenebilir (güncelleme = sil + ekle).

use rkyv::rancor::Error as RkyvHata;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::path::Path;

/// Dosya biçimi sürümü; düzen değişince artar, eski dosya açık hatayla reddedilir.
pub const SURUM: u32 = 1;
const IMZA: &[u8; 4] = b"BGED";
const BASLIK: usize = 16;

#[derive(Debug)]
pub enum Hata {
    Boyut { beklenen: usize, gelen: usize },
    GecersizVektor(String),
    Yineleme(u64),
    Dosya { yol: String, neden: String },
    Bicim(String),
}

impl fmt::Display for Hata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Hata::Boyut { beklenen, gelen } => {
                write!(f, "vektör boyutu {gelen}, dizinin boyutu {beklenen}")
            }
            Hata::GecersizVektor(n) => write!(f, "geçersiz vektör: {n}"),
            Hata::Yineleme(k) => write!(f, "kimlik {k} dizinde zaten var (önce silin)"),
            Hata::Dosya { yol, neden } => write!(f, "{yol}: {neden}"),
            Hata::Bicim(n) => write!(f, "dizin dosyası geçersiz: {n}"),
        }
    }
}

impl std::error::Error for Hata {}

pub type Sonuc<T> = Result<T, Hata>;

/// HNSW ayarları. `m`: düğüm başına komşu (0. katmanda 2m); `ef_insa`: eklemede aday
/// kümesi genişliği. Büyüdükçe kurulum yavaşlar, isabet artar.
#[derive(Clone, Copy, Debug, PartialEq, Archive, Serialize, Deserialize)]
pub struct Ayar {
    pub m: u32,
    pub ef_insa: u32,
    pub tohum: u64,
}

impl Default for Ayar {
    fn default() -> Self {
        Ayar {
            m: 16,
            ef_insa: 100,
            tohum: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Yakınlık dizini. İç numara (`u32`) ekleme sırasıdır; dış kimlik (`u64`) çağıranındır
/// (ör. veritabanı satır kimliği).
#[derive(Archive, Serialize, Deserialize)]
pub struct Dizin {
    boyut: u32,
    ayar: Ayar,
    kimlik: Vec<u64>,
    vektor: Vec<f32>,
    /// `komsu[düğüm][katman]`; düğümün katman sayısı kendi seviyesi + 1.
    komsu: Vec<Vec<Vec<u32>>>,
    mezar: Vec<bool>,
    /// Canlı kimlik → iç numara.
    canli: HashMap<u64, u32>,
    giris: Option<u32>,
    ust: u32,
    rng: u64,
}

/// Aday: uzaklık (1 − kosinüs) ve iç numara. Sıralama uzaklığa, eşitlikte numaraya göre.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Aday {
    uz: f32,
    no: u32,
}

impl Eq for Aday {}
impl Ord for Aday {
    fn cmp(&self, o: &Self) -> Ordering {
        self.uz.total_cmp(&o.uz).then(self.no.cmp(&o.no))
    }
}
impl PartialOrd for Aday {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Sekiz şeritli iç çarpım: derleyici SIMD'e çevirebilsin diye kısmi toplamlar ayrı.
fn ic_carpim(a: &[f32], b: &[f32]) -> f32 {
    let mut t = [0f32; 8];
    let mut ka = a.chunks_exact(8);
    let mut kb = b.chunks_exact(8);
    for (x, y) in ka.by_ref().zip(kb.by_ref()) {
        for i in 0..8 {
            t[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = t.iter().sum();
    for (x, y) in ka.remainder().iter().zip(kb.remainder()) {
        s += x * y;
    }
    s
}

/// L2 normalize eder; sonlu olmayan ya da sıfır vektörü reddeder.
fn normalize(v: &[f32]) -> Sonuc<Vec<f32>> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(Hata::GecersizVektor(format!("{i}. bileşen sonlu değil")));
    }
    let n = ic_carpim(v, v).sqrt();
    if n == 0.0 || !n.is_finite() {
        return Err(Hata::GecersizVektor("uzunluk sıfır".into()));
    }
    Ok(v.iter().map(|x| x / n).collect())
}

/// Ziyaret kümesi: düğüm başına bir bit.
struct Ziyaret(Vec<u64>);
impl Ziyaret {
    fn yeni(n: usize) -> Self {
        Ziyaret(vec![0; n.div_ceil(64)])
    }
    /// İlk ziyaretse true.
    fn isaretle(&mut self, no: u32) -> bool {
        let (k, b) = ((no / 64) as usize, 1u64 << (no % 64));
        let ilk = self.0[k] & b == 0;
        self.0[k] |= b;
        ilk
    }
}

impl Dizin {
    pub fn yeni(boyut: usize, ayar: Ayar) -> Sonuc<Dizin> {
        if boyut == 0 || ayar.m < 2 || ayar.ef_insa == 0 {
            return Err(Hata::Bicim(format!(
                "boyut {boyut}, m {}, ef_insa {}: boyut > 0, m ≥ 2, ef_insa > 0 olmalı",
                ayar.m, ayar.ef_insa
            )));
        }
        Ok(Dizin {
            boyut: boyut as u32,
            ayar,
            kimlik: Vec::new(),
            vektor: Vec::new(),
            komsu: Vec::new(),
            mezar: Vec::new(),
            canli: HashMap::new(),
            giris: None,
            ust: 0,
            rng: ayar.tohum | 1,
        })
    }

    pub fn boyut(&self) -> usize {
        self.boyut as usize
    }

    /// Canlı (silinmemiş) vektör sayısı.
    pub fn canli_sayisi(&self) -> usize {
        self.canli.len()
    }

    /// Mezar taşları dahil düğüm sayısı.
    pub fn dugum_sayisi(&self) -> usize {
        self.kimlik.len()
    }

    pub fn icerir(&self, kimlik: u64) -> bool {
        self.canli.contains_key(&kimlik)
    }

    fn vek(&self, no: u32) -> &[f32] {
        let b = self.boyut as usize;
        &self.vektor[no as usize * b..(no as usize + 1) * b]
    }

    fn uz(&self, q: &[f32], no: u32) -> f32 {
        1.0 - ic_carpim(q, self.vek(no))
    }

    /// Seviye: ⌊−ln(U)·mL⌋, mL = 1/ln(m). xorshift64*, tohumdan belirlenimli.
    fn rastgele_seviye(&mut self) -> u32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let r = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let u = ((r >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let ml = 1.0 / (self.ayar.m as f64).ln();
        ((-u.ln() * ml).floor() as u32).min(16)
    }

    /// Bir katmanda açgözlü iniş: daha yakın komşu kalmayınca durur.
    fn acgozlu(&self, q: &[f32], mut en: Aday, katman: usize) -> Aday {
        loop {
            let mut degisti = false;
            for &n in &self.komsu[en.no as usize][katman] {
                let d = self.uz(q, n);
                if d < en.uz {
                    en = Aday { uz: d, no: n };
                    degisti = true;
                }
            }
            if !degisti {
                return en;
            }
        }
    }

    /// Katman araması (makalenin 2. algoritması): en yakın `ef` aday, artan uzaklıkla.
    fn katman_ara(&self, q: &[f32], girisler: &[Aday], ef: usize, katman: usize) -> Vec<Aday> {
        let mut ziyaret = Ziyaret::yeni(self.kimlik.len());
        let mut adaylar: BinaryHeap<std::cmp::Reverse<Aday>> = BinaryHeap::new();
        let mut sonuc: BinaryHeap<Aday> = BinaryHeap::new();
        for &g in girisler {
            ziyaret.isaretle(g.no);
            adaylar.push(std::cmp::Reverse(g));
            sonuc.push(g);
        }
        while sonuc.len() > ef {
            sonuc.pop();
        }
        while let Some(std::cmp::Reverse(c)) = adaylar.pop() {
            let en_uzak = sonuc.peek().map_or(f32::INFINITY, |a| a.uz);
            if c.uz > en_uzak && sonuc.len() >= ef {
                break;
            }
            for &n in &self.komsu[c.no as usize][katman] {
                if !ziyaret.isaretle(n) {
                    continue;
                }
                let d = self.uz(q, n);
                let en_uzak = sonuc.peek().map_or(f32::INFINITY, |a| a.uz);
                if sonuc.len() < ef || d < en_uzak {
                    let a = Aday { uz: d, no: n };
                    adaylar.push(std::cmp::Reverse(a));
                    sonuc.push(a);
                    if sonuc.len() > ef {
                        sonuc.pop();
                    }
                }
            }
        }
        sonuc.into_sorted_vec()
    }

    /// Çeşitlilik sezgisi: aday, seçilmişlerin hepsine sorgudan daha uzaksa alınır;
    /// eksik kalan yer budananlarla doldurulur. `adaylar` artan uzaklıkta.
    fn sec(&self, adaylar: &[Aday], m: usize) -> Vec<u32> {
        let mut secilen: Vec<Aday> = Vec::with_capacity(m);
        let mut budanan: Vec<u32> = Vec::new();
        for &c in adaylar {
            if secilen.len() >= m {
                break;
            }
            let cv = self.vek(c.no);
            if secilen.iter().all(|s| self.uz(cv, s.no) > c.uz) {
                secilen.push(c);
            } else {
                budanan.push(c.no);
            }
        }
        let mut sonuc: Vec<u32> = secilen.iter().map(|a| a.no).collect();
        for b in budanan {
            if sonuc.len() >= m {
                break;
            }
            sonuc.push(b);
        }
        sonuc
    }

    /// Ekler. Kimlik canlıysa ret (önce `sil`); vektör normalize edilir.
    pub fn ekle(&mut self, kimlik: u64, v: &[f32]) -> Sonuc<()> {
        if v.len() != self.boyut as usize {
            return Err(Hata::Boyut {
                beklenen: self.boyut as usize,
                gelen: v.len(),
            });
        }
        if self.canli.contains_key(&kimlik) {
            return Err(Hata::Yineleme(kimlik));
        }
        if self.kimlik.len() >= u32::MAX as usize {
            return Err(Hata::Bicim("düğüm sayısı u32 sınırında".into()));
        }
        let q = normalize(v)?;
        let no = self.kimlik.len() as u32;
        let seviye = self.rastgele_seviye();
        self.kimlik.push(kimlik);
        self.vektor.extend_from_slice(&q);
        self.komsu.push(vec![Vec::new(); seviye as usize + 1]);
        self.mezar.push(false);
        self.canli.insert(kimlik, no);

        let Some(giris) = self.giris else {
            self.giris = Some(no);
            self.ust = seviye;
            return Ok(());
        };
        let mut en = Aday {
            uz: self.uz(&q, giris),
            no: giris,
        };
        for katman in (seviye + 1..=self.ust).rev() {
            en = self.acgozlu(&q, en, katman as usize);
        }
        let m = self.ayar.m as usize;
        let mut girisler = vec![en];
        for katman in (0..=seviye.min(self.ust)).rev() {
            let k = katman as usize;
            let w = self.katman_ara(&q, &girisler, self.ayar.ef_insa as usize, k);
            let secilen = self.sec(&w, m);
            let mmax = if k == 0 { 2 * m } else { m };
            for &n in &secilen {
                self.komsu[n as usize][k].push(no);
                if self.komsu[n as usize][k].len() > mmax {
                    let nv = self.vek(n).to_vec();
                    let mut a: Vec<Aday> = self.komsu[n as usize][k]
                        .iter()
                        .map(|&x| Aday {
                            uz: self.uz(&nv, x),
                            no: x,
                        })
                        .collect();
                    a.sort();
                    self.komsu[n as usize][k] = self.sec(&a, mmax);
                }
            }
            self.komsu[no as usize][k] = secilen;
            girisler = w;
        }
        if seviye > self.ust {
            self.giris = Some(no);
            self.ust = seviye;
        }
        Ok(())
    }

    /// Siler (mezar taşı). Kimlik yoksa false.
    pub fn sil(&mut self, kimlik: u64) -> bool {
        match self.canli.remove(&kimlik) {
            Some(no) => {
                self.mezar[no as usize] = true;
                true
            }
            None => false,
        }
    }

    /// En benzer `k` canlı vektör: (kimlik, kosinüs benzerliği), azalan benzerlikle.
    /// `ef` aday kümesidir (en az `k` alınır); büyüdükçe isabet artar, hız düşer.
    pub fn ara(&self, sorgu: &[f32], k: usize, ef: usize) -> Sonuc<Vec<(u64, f32)>> {
        if sorgu.len() != self.boyut as usize {
            return Err(Hata::Boyut {
                beklenen: self.boyut as usize,
                gelen: sorgu.len(),
            });
        }
        let q = normalize(sorgu)?;
        let Some(giris) = self.giris else {
            return Ok(Vec::new());
        };
        let mut en = Aday {
            uz: self.uz(&q, giris),
            no: giris,
        };
        for katman in (1..=self.ust).rev() {
            en = self.acgozlu(&q, en, katman as usize);
        }
        let w = self.katman_ara(&q, &[en], ef.max(k), 0);
        Ok(w.into_iter()
            .filter(|a| !self.mezar[a.no as usize])
            .take(k)
            .map(|a| (self.kimlik[a.no as usize], 1.0 - a.uz))
            .collect())
    }

    /// Başlıklı rkyv dosyasına yazar: geçici dosya + yeniden adlandırma (yarım dosya kalmaz).
    pub fn kaydet(&self, yol: &Path) -> Sonuc<()> {
        let dosya = |neden: String| Hata::Dosya {
            yol: yol.display().to_string(),
            neden,
        };
        let yuk =
            rkyv::to_bytes::<RkyvHata>(self).map_err(|e| dosya(format!("kodlanamadı: {e}")))?;
        let mut veri = Vec::with_capacity(BASLIK + yuk.len());
        veri.extend_from_slice(IMZA);
        veri.extend_from_slice(&SURUM.to_le_bytes());
        veri.extend_from_slice(&[0u8; 8]);
        veri.extend_from_slice(&yuk);
        let gecici = yol.with_extension("yaziliyor");
        std::fs::write(&gecici, &veri).map_err(|e| dosya(format!("yazılamadı: {e}")))?;
        std::fs::rename(&gecici, yol).map_err(|e| dosya(format!("taşınamadı: {e}")))
    }

    /// Okur, doğrular ve tutarlılığını denetler.
    pub fn yukle(yol: &Path) -> Sonuc<Dizin> {
        let ad = yol.display().to_string();
        let veri = std::fs::read(yol).map_err(|e| Hata::Dosya {
            yol: ad.clone(),
            neden: format!("okunamadı: {e}"),
        })?;
        Dizin::baytlardan(&veri).map_err(|e| match e {
            Hata::Bicim(n) => Hata::Bicim(format!("{ad}: {n}")),
            o => o,
        })
    }

    fn baytlardan(veri: &[u8]) -> Sonuc<Dizin> {
        let Some((bas, yuk)) = veri.split_at_checked(BASLIK) else {
            return Err(Hata::Bicim(format!("dosya kısa ({} bayt)", veri.len())));
        };
        if &bas[..4] != IMZA {
            return Err(Hata::Bicim(
                "imza tanınmadı (bge-dizin dosyası değil)".into(),
            ));
        }
        let s = u32::from_le_bytes([bas[4], bas[5], bas[6], bas[7]]);
        if s != SURUM {
            return Err(Hata::Bicim(format!(
                "biçim sürümü {s} ≠ {SURUM}; dizini yeniden kurun"
            )));
        }
        let mut hizali = AlignedVec::<16>::with_capacity(yuk.len());
        hizali.extend_from_slice(yuk);
        let d = rkyv::from_bytes::<Dizin, RkyvHata>(&hizali)
            .map_err(|e| Hata::Bicim(format!("doğrulanamadı (bozuk): {e}")))?;
        d.tutarlilik()?;
        Ok(d)
    }

    /// Bayt düzeyi doğru ama anlamca bozuk dosyayı yakalar (ör. sınır dışı komşu).
    fn tutarlilik(&self) -> Sonuc<()> {
        let n = self.kimlik.len();
        let hata = |n: &str| Err(Hata::Bicim(format!("tutarsız: {n}")));
        if self.boyut == 0 || self.vektor.len() != n * self.boyut as usize {
            return hata("vektör uzunluğu");
        }
        if self.komsu.len() != n || self.mezar.len() != n {
            return hata("düğüm sayıları");
        }
        if self.giris.map_or(n != 0, |g| g as usize >= n) {
            return hata("giriş noktası");
        }
        for k in &self.komsu {
            if k.is_empty() || k.len() > self.ust as usize + 1 {
                return hata("katman sayısı");
            }
            if k.iter().flatten().any(|&x| x as usize >= n) {
                return hata("sınır dışı komşu");
            }
        }
        for (&kim, &no) in &self.canli {
            if no as usize >= n || self.kimlik[no as usize] != kim || self.mezar[no as usize] {
                return hata("canlı kimlik eşlemi");
            }
        }
        if self.mezar.iter().filter(|m| !**m).count() != self.canli.len() {
            return hata("canlı sayısı");
        }
        Ok(())
    }
}

#[cfg(test)]
mod testler {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f32 {
            let (a, b) = (self.u().max(1e-12), self.u());
            ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
        }
    }

    /// Kümelenmiş sentetik veri (gerçek gömmeler gibi yapılı; düzgün dağılım değil).
    fn kumeli(n: usize, d: usize, kume: usize, tohum: u64) -> Vec<Vec<f32>> {
        let mut r = Rng(tohum);
        let merkez: Vec<Vec<f32>> = (0..kume)
            .map(|_| (0..d).map(|_| r.normal()).collect())
            .collect();
        (0..n)
            .map(|i| {
                merkez[i % kume]
                    .iter()
                    .map(|c| c + 0.35 * r.normal())
                    .collect()
            })
            .collect()
    }

    fn kaba(veri: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u64> {
        let qn = normalize(q).unwrap();
        let mut v: Vec<(f32, u64)> = veri
            .iter()
            .enumerate()
            .map(|(i, x)| (ic_carpim(&qn, &normalize(x).unwrap()), i as u64))
            .collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        v.into_iter().take(k).map(|x| x.1).collect()
    }

    fn kur(veri: &[Vec<f32>]) -> Dizin {
        let mut d = Dizin::yeni(veri[0].len(), Ayar::default()).unwrap();
        for (i, v) in veri.iter().enumerate() {
            d.ekle(i as u64, v).unwrap();
        }
        d
    }

    fn isabet(d: &Dizin, veri: &[Vec<f32>], sorgular: &[Vec<f32>], ef: usize) -> f64 {
        let mut toplam = 0usize;
        for q in sorgular {
            let dogru = kaba(veri, q, 10);
            let bulunan: Vec<u64> = d.ara(q, 10, ef).unwrap().into_iter().map(|x| x.0).collect();
            toplam += dogru.iter().filter(|x| bulunan.contains(x)).count();
        }
        toplam as f64 / (sorgular.len() * 10) as f64
    }

    #[test]
    fn isabet_kaba_kuvvete_yakin() {
        // Sorgular aynı dağılımdan, dizine konmayan ayrık örnek (aynı merkezler, yeni
        // gürültü). Başka tohumun merkezleri dağılım dışı sorgu olurdu (bkz. README).
        let tum = kumeli(3060, 64, 30, 7);
        let (veri, sorgular) = tum.split_at(3000);
        let d = kur(veri);
        let r = isabet(&d, veri, sorgular, 64);
        assert!(r >= 0.95, "recall@10 = {r}");
        // Benzerlikler kesin: ilk sonucun benzerliği kaba kuvvetinkiyle aynı.
        let q = &sorgular[0];
        let ilk = d.ara(q, 1, 64).unwrap()[0];
        let dogru = ic_carpim(
            &normalize(q).unwrap(),
            &normalize(&veri[ilk.0 as usize]).unwrap(),
        );
        assert!((ilk.1 - dogru).abs() < 1e-5);
    }

    #[test]
    fn kaydet_yukle_ayni_sonuc() {
        let veri = kumeli(800, 32, 10, 3);
        let d = kur(&veri);
        let yol = std::env::temp_dir().join(format!("bge-dizin-{}-ky.bged", std::process::id()));
        d.kaydet(&yol).unwrap();
        let y = Dizin::yukle(&yol).unwrap();
        for q in kumeli(20, 32, 10, 5) {
            assert_eq!(d.ara(&q, 10, 50).unwrap(), y.ara(&q, 10, 50).unwrap());
        }
        assert_eq!(y.canli_sayisi(), 800);
        std::fs::remove_file(&yol).unwrap();
    }

    #[test]
    fn bozuk_dosya_acik_hata() {
        let d = kur(&kumeli(200, 16, 5, 1));
        let yol = std::env::temp_dir().join(format!("bge-dizin-{}-bz.bged", std::process::id()));
        d.kaydet(&yol).unwrap();
        let saglam = std::fs::read(&yol).unwrap();
        let ret = |b: &[u8]| match Dizin::baytlardan(b) {
            Err(Hata::Bicim(m)) => m,
            Err(e) => panic!("beklenmeyen hata türü: {e}"),
            Ok(_) => panic!("bozuk dosya kabul edildi"),
        };
        assert!(ret(&saglam[..10]).contains("kısa"));
        let mut b = saglam.clone();
        b[0] = b'X';
        assert!(ret(&b).contains("imza"));
        let mut b = saglam.clone();
        b[4] = 9;
        assert!(ret(&b).contains("sürüm"));
        let mut b = saglam.clone();
        b.truncate(saglam.len() - 7);
        ret(&b);
        // Yükün ortasındaki baytlar: ya rkyv doğrulaması ya tutarlılık denetimi
        // reddetmeli ya da (bir vektör bileşeni değiştiyse) yükleme yine tutarlı olmalı.
        // Komşu tablosunu sınır dışına iten bozulmayı açıkça sına:
        let mut d2 = kur(&kumeli(50, 8, 2, 4));
        d2.komsu[0][0].push(10_000);
        d2.kaydet(&yol).unwrap();
        assert!(ret(&std::fs::read(&yol).unwrap()).contains("sınır dışı komşu"));
        std::fs::remove_file(&yol).unwrap();
    }

    #[test]
    fn mezar_ve_yeniden_ekleme() {
        let veri = kumeli(500, 16, 5, 2);
        let mut d = kur(&veri);
        let q = &veri[17];
        assert_eq!(d.ara(q, 1, 32).unwrap()[0].0, 17);
        assert!(d.sil(17));
        assert!(!d.sil(17));
        assert!(d.ara(q, 10, 64).unwrap().iter().all(|x| x.0 != 17));
        assert_eq!(d.canli_sayisi(), 499);
        d.ekle(17, q).unwrap();
        assert_eq!(d.ara(q, 1, 32).unwrap()[0].0, 17);
        assert_eq!(d.dugum_sayisi(), 501);
    }

    #[test]
    fn gecersiz_girdiler_reddedilir() {
        let mut d = Dizin::yeni(4, Ayar::default()).unwrap();
        assert!(matches!(d.ekle(1, &[1.0, 0.0]), Err(Hata::Boyut { .. })));
        assert!(matches!(d.ekle(1, &[0.0; 4]), Err(Hata::GecersizVektor(_))));
        assert!(matches!(
            d.ekle(1, &[f32::NAN, 1.0, 0.0, 0.0]),
            Err(Hata::GecersizVektor(_))
        ));
        d.ekle(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            d.ekle(1, &[0.0, 1.0, 0.0, 0.0]),
            Err(Hata::Yineleme(1))
        ));
        assert!(Dizin::yeni(0, Ayar::default()).is_err());
        assert!(
            Dizin::yeni(
                4,
                Ayar {
                    m: 1,
                    ..Ayar::default()
                }
            )
            .is_err()
        );
        let bos = Dizin::yeni(4, Ayar::default()).unwrap();
        assert!(bos.ara(&[1.0, 0.0, 0.0, 0.0], 5, 10).unwrap().is_empty());
        assert_eq!(d.ara(&[1.0, 0.0, 0.0, 0.0], 5, 10).unwrap().len(), 1);
    }
}
