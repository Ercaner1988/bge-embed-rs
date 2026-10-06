//! bge-dizin ölçümü: kurulum süresi, recall@10 (kaba kuvvete göre), arama p50/p95.
//!
//! Sentetik (kümelenmiş, gerçek gömmeleri temsil etmez):
//!   cargo run --release -p bge-dizin --example olc -- sentetik 100000 1024 [küme=200]
//! Gerçek (ham f32 LE dosyası, satır başına `boyut` sayı; ör. Turso'dan dökülmüş bge-m3):
//!   cargo run --release -p bge-dizin --example olc -- gercek <dosya> 1024
//! Sorgular dizine konmayan ayrık bir örnektir (her 200 vektörden biri, en çok 200).
//! Kabul eşiği (yol haritası Görev 2): recall@10 ≥ 0,95 ve p50 < 20 ms.

use bge_dizin::{Ayar, Dizin};
use std::time::Instant;

fn rng(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn normal(s: &mut u64) -> f32 {
    let (a, b) = (rng(s).max(1e-12), rng(s));
    ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
}

fn kumeli(n: usize, d: usize, kume: usize) -> Vec<Vec<f32>> {
    let mut s = 0x1234_5678_9abc_def1u64;
    let merkez: Vec<Vec<f32>> = (0..kume)
        .map(|_| (0..d).map(|_| normal(&mut s)).collect())
        .collect();
    (0..n)
        .map(|i| {
            merkez[i % kume]
                .iter()
                .map(|c| c + 0.35 * normal(&mut s))
                .collect()
        })
        .collect()
}

fn dosyadan(yol: &str, d: usize) -> Result<Vec<Vec<f32>>, String> {
    let b = std::fs::read(yol).map_err(|e| format!("{yol}: {e}"))?;
    if b.len() % (4 * d) != 0 {
        return Err(format!(
            "{yol}: {} bayt, {d} boyutlu f32 satırlarına bölünmüyor",
            b.len()
        ));
    }
    Ok(b.chunks_exact(4 * d)
        .map(|r| {
            r.chunks_exact(4)
                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect()
        })
        .collect())
}

fn norm(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn yuzdelik(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (kip, veri) = match a.first().map(String::as_str) {
        Some("sentetik") => {
            let n = a.get(1).and_then(|x| x.parse().ok()).unwrap_or(100_000);
            let d = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(1024);
            let k = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(200);
            (format!("SENTETİK (kümelenmiş, {k} küme)"), kumeli(n, d, k))
        }
        Some("gercek") => {
            let (Some(yol), Some(d)) = (a.get(1), a.get(2).and_then(|x| x.parse().ok())) else {
                eprintln!("kullanım: olc gercek <dosya> <boyut>");
                std::process::exit(2);
            };
            match dosyadan(yol, d) {
                Ok(v) => (format!("GERÇEK ({yol})"), v),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("kullanım: olc sentetik [n] [boyut] [küme] | olc gercek <dosya> <boyut>");
            std::process::exit(2);
        }
    };
    let d = veri[0].len();
    let (mut sorgu, mut dizinlik) = (Vec::new(), Vec::new());
    for (i, v) in veri.into_iter().enumerate() {
        if i % 200 == 7 && sorgu.len() < 200 {
            sorgu.push(v);
        } else {
            dizinlik.push(v);
        }
    }
    println!(
        "veri: {kip}; dizine {} × {d}, sorgu {}",
        dizinlik.len(),
        sorgu.len()
    );

    let t = Instant::now();
    let mut dz = match Dizin::yeni(d, Ayar::default()) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    for (i, v) in dizinlik.iter().enumerate() {
        if let Err(e) = dz.ekle(i as u64, v) {
            eprintln!("ekleme {i}: {e}");
            std::process::exit(1);
        }
    }
    println!(
        "kurulum: {:.1} s (tek iş parçacığı, {:?})",
        t.elapsed().as_secs_f64(),
        Ayar::default()
    );

    // Kaba kuvvet: kesin ilk 10 ve tam tarama süresi (karşılaştırma için).
    let normlu: Vec<Vec<f32>> = dizinlik.iter().map(|v| norm(v)).collect();
    let mut kaba_ms = Vec::new();
    let dogru: Vec<Vec<u64>> = sorgu
        .iter()
        .map(|q| {
            let t = Instant::now();
            let qn = norm(q);
            let mut s: Vec<(f32, u64)> = normlu
                .iter()
                .enumerate()
                .map(|(i, v)| (v.iter().zip(&qn).map(|(a, b)| a * b).sum(), i as u64))
                .collect();
            s.sort_by(|a, b| b.0.total_cmp(&a.0));
            kaba_ms.push(t.elapsed().as_secs_f64() * 1e3);
            s.into_iter().take(10).map(|x| x.1).collect()
        })
        .collect();
    println!(
        "kaba kuvvet (bellekte, Rust): p50 {:.2} ms",
        yuzdelik(&mut kaba_ms, 0.5)
    );

    for ef in [32usize, 64, 128, 256] {
        let mut ms = Vec::new();
        let mut isabet = 0usize;
        for (q, dg) in sorgu.iter().zip(&dogru) {
            let t = Instant::now();
            let r = match dz.ara(q, 10, ef) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("arama: {e}");
                    std::process::exit(1);
                }
            };
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            isabet += dg.iter().filter(|x| r.iter().any(|y| y.0 == **x)).count();
        }
        let recall = isabet as f64 / (sorgu.len() * 10) as f64;
        let p50 = yuzdelik(&mut ms, 0.5);
        let p95 = yuzdelik(&mut ms, 0.95);
        let kabul = recall >= 0.95 && p50 < 20.0;
        println!(
            "ef {ef:>3}: recall@10 {recall:.4}  p50 {p50:.2} ms  p95 {p95:.2} ms  {}",
            if kabul { "KABUL" } else { "—" }
        );
    }

    let yol = std::env::temp_dir().join(format!("bge-dizin-olc-{}.bged", std::process::id()));
    let t = Instant::now();
    match dz.kaydet(&yol) {
        Ok(()) => {
            let boy = std::fs::metadata(&yol).map(|m| m.len()).unwrap_or(0);
            let yaz = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let yuk = Dizin::yukle(&yol).map(|y| y.canli_sayisi());
            println!(
                "dosya: {:.1} MiB, yazma {yaz:.2} s, yükleme {:.2} s ({yuk:?} canlı)",
                boy as f64 / 1048576.0,
                t.elapsed().as_secs_f64()
            );
            if let Err(e) = std::fs::remove_file(&yol) {
                eprintln!("geçici dosya silinemedi: {e}");
            }
        }
        Err(e) => eprintln!("kaydet: {e}"),
    }
}
