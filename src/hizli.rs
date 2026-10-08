// CPU çekirdekleri. candle'ın CPU yolunda GELU/softmax/yan-eklemeler tek iş parçacıklı ve
// k=1024 matmul'ı küçük m'de (arama sorgusu: 5-10 token) bellek bandını kullanamıyor.
// Ölçüm (Ryzen 5 7430U, 12 mantıksal çekirdek): 200 token'lık girdide katman başına
// ~92 ms'nin ~22 ms'si tek iş parçacıklı GELU (10 ms) + softmax (11 ms); 6 token'da
// matmul'lar ~4 GFLOPS'ta kalıyor. Burada hepsi rayon ile paralel ve birleştirilmiş.
// Sayısal sözleşme: aynı erf (libm::erff), aynı LayerNorm formülü; yalnız toplama sırası
// değişir (cosine ≥ 0.999 doğrulaması main.rs testlerinde ve el ölçümünde).
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, Layout, Result, Shape, Tensor, bail, cpu::erf::erf_f32,
};
use half::{f16, slice::HalfFloatSliceExt};
use rayon::prelude::*;

/// Bu satır sayısının altında matmul yerine `KucukLinear` kullanılır (ölçümle seçildi).
pub const KUCUK_M: usize = 32;

fn f32_dilim<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let v = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&v[a..b]),
        None => bail!("hizli: tensör bitişik olmalı"),
    }
}

fn f16_dilim<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f16]> {
    let v = s.as_slice::<f16>()?;
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&v[a..b]),
        None => bail!("hizli: tensör bitişik olmalı"),
    }
}

/// Ağırlık saklama biçimi değişimi (f32 ⇄ f16), rayon ile paralel, F16C ile vektörel.
/// candle'ın `to_dtype`'u tek iş parçacıklı ve öğe öğe: 568 M ağırlıkta saniyeler sürer.
pub struct F16e;
pub struct F32ye;
const DONUSUM_PARCA: usize = 1 << 16;

impl CustomOp1 for F16e {
    fn name(&self) -> &'static str {
        "f16e"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let x = f32_dilim(s, l)?;
        let mut y = vec![f16::ZERO; x.len()];
        y.par_chunks_mut(DONUSUM_PARCA)
            .zip(x.par_chunks(DONUSUM_PARCA))
            .for_each(|(o, i)| o.convert_from_f32_slice(i));
        Ok((CpuStorage::F16(y), l.shape().clone()))
    }
}

impl CustomOp1 for F32ye {
    fn name(&self) -> &'static str {
        "f32ye"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let x = f16_dilim(s, l)?;
        let mut y = vec![0f32; x.len()];
        y.par_chunks_mut(DONUSUM_PARCA)
            .zip(x.par_chunks(DONUSUM_PARCA))
            .for_each(|(o, i)| i.convert_to_f32_slice(o));
        Ok((CpuStorage::F32(y), l.shape().clone()))
    }
}

/// 32 şeritli FMA nokta çarpımı; LLVM bunu 4 ymm biriktiriciye açar.
#[inline(always)]
fn nokta(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let ((ac, ar), (bc, br)) = (a[..n].as_chunks::<32>(), b[..n].as_chunks::<32>());
    let mut acc = [0f32; 32];
    for (x, y) in ac.iter().zip(bc) {
        for l in 0..32 {
            acc[l] = x[l].mul_add(y[l], acc[l]);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ar.iter().zip(br) {
        s = x.mul_add(*y, s);
    }
    s
}

/// y = x · wᵀ, x (m,k) f32, w (n,k) f16 bitişik. W satırı bir kez bellekten okunur, f32'ye açılıp
/// m satırla L1'den yeniden kullanılır; n üzerinden paralel. Küçük m'de iş bellek bandına
/// bağlı: f16 ağırlık okunan baytı yarıya indirir.
pub struct KucukLinear;

impl CustomOp2 for KucukLinear {
    fn name(&self) -> &'static str {
        "kucuk-linear"
    }
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, w) = (f32_dilim(s1, l1)?, f16_dilim(s2, l2)?);
        let (m, k) = l1.shape().dims2()?;
        let n = l2.shape().dims2()?.0;
        const SATIR: usize = 16;
        // Çıktı j-büyük (n,m) yazılır ki paralel parçalar ayrık kalsın; sonda devrilir.
        let mut yt = vec![0f32; n * m];
        yt.par_chunks_mut(SATIR * m)
            .enumerate()
            .for_each(|(c, out)| {
                let mut wj = vec![0f32; k];
                for (jj, o) in out.chunks_mut(m).enumerate() {
                    w[(c * SATIR + jj) * k..][..k].convert_to_f32_slice(&mut wj);
                    for (r, v) in o.iter_mut().enumerate() {
                        *v = nokta(&x[r * k..][..k], &wj);
                    }
                }
            });
        let mut y = vec![0f32; m * n];
        for j in 0..n {
            for r in 0..m {
                y[r * n + j] = yt[j * m + r];
            }
        }
        Ok((CpuStorage::F32(y), Shape::from_dims(&[m, n])))
    }
}

/// gelu_erf(x + bias), satır uzunluğu bias.len(); rayon ile paralel.
pub struct BiasGelu<'a>(pub &'a [f32]);

impl CustomOp1 for BiasGelu<'_> {
    fn name(&self) -> &'static str {
        "bias-gelu"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let x = f32_dilim(s, l)?;
        let n = self.0.len();
        let mut y = vec![0f32; x.len()];
        y.par_chunks_mut(n).zip(x.par_chunks(n)).for_each(|(o, i)| {
            for ((o, &v), &b) in o.iter_mut().zip(i).zip(self.0) {
                let v = v + b;
                *o = (erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v;
            }
        });
        Ok((CpuStorage::F32(y), l.shape().clone()))
    }
}

/// LayerNorm(x + bias + artık): BERT'in "Dense → +girdi → LayerNorm" kuyruğu tek geçişte.
pub struct ArtikLn<'a> {
    pub bias: &'a [f32],
    pub agirlik: &'a [f32],
    pub kayma: &'a [f32],
    pub eps: f32,
}

impl CustomOp2 for ArtikLn<'_> {
    fn name(&self) -> &'static str {
        "artik-ln"
    }
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, r) = (f32_dilim(s1, l1)?, f32_dilim(s2, l2)?);
        let h = self.bias.len();
        let mut y = vec![0f32; x.len()];
        y.par_chunks_mut(h)
            .zip(x.par_chunks(h).zip(r.par_chunks(h)))
            .for_each(|(o, (xr, rr))| {
                for (((o, &a), &b), &c) in o.iter_mut().zip(xr).zip(rr).zip(self.bias) {
                    *o = a + b + c;
                }
                let ort = o.iter().sum::<f32>() / h as f32;
                let var = o.iter().map(|v| (v - ort) * (v - ort)).sum::<f32>() / h as f32;
                let ters = 1.0 / (var + self.eps).sqrt();
                for ((o, &g), &b) in o.iter_mut().zip(self.agirlik).zip(self.kayma) {
                    *o = (*o - ort) * ters * g + b;
                }
            });
        Ok((CpuStorage::F32(y), l1.shape().clone()))
    }
}

/// Çok başlı öz-dikkat: girdi birleşik QKV (B·L, 3·H) (q|k|v yan yana, bias EKLENMEMİŞ),
/// çıktı (B·L, H). Sağdan doldurulmuş partiye uygun: satır b'nin anahtarları yalnız
/// `uzunluk[b]` kadardır. Ölçek (1/√64 = 1/8) q'ya katılır; transpoze/bitişik kopya yok,
/// L×L skor matrisi hiç oluşmaz (satır başına O(L) bellek).
pub struct Dikkat<'a> {
    pub bias: &'a [f32],
    pub uzunluk: &'a [usize],
    pub bas: usize,
    pub bas_boyu: usize,
    pub l: usize,
}

impl CustomOp1 for Dikkat<'_> {
    fn name(&self) -> &'static str {
        "dikkat"
    }
    fn cpu_fwd(&self, s: &CpuStorage, lay: &Layout) -> Result<(CpuStorage, Shape)> {
        let qkv = f32_dilim(s, lay)?;
        let (bas, d, l) = (self.bas, self.bas_boyu, self.l);
        let h = bas * d;
        let ks = 3 * h;
        let olcek = 1.0 / (d as f32).sqrt();
        const BLOK: usize = 64;
        let gorevler: Vec<(usize, usize, usize)> = (0..self.uzunluk.len())
            .flat_map(|b| {
                (0..bas).flat_map(move |hd| (0..l.div_ceil(BLOK)).map(move |blk| (b, hd, blk)))
            })
            .collect();
        let parcalar: Vec<Vec<f32>> = gorevler
            .par_iter()
            .map(|&(b, hd, blk)| {
                let len = self.uzunluk[b];
                let (i0, i1) = (blk * BLOK, ((blk + 1) * BLOK).min(l).min(len));
                if i0 >= i1 {
                    return Vec::new();
                }
                let satir = |i: usize, kayma: usize| &qkv[(b * l + i) * ks + kayma + hd * d..][..d];
                let (bq, bk, bv) = (
                    &self.bias[hd * d..][..d],
                    &self.bias[h + hd * d..][..d],
                    &self.bias[2 * h + hd * d..][..d],
                );
                let mut kh = Vec::with_capacity(len * d);
                let mut vh = Vec::with_capacity(len * d);
                for j in 0..len {
                    kh.extend(satir(j, h).iter().zip(bk).map(|(a, b)| a + b));
                    vh.extend(satir(j, 2 * h).iter().zip(bv).map(|(a, b)| a + b));
                }
                let mut cikti = vec![0f32; (i1 - i0) * d];
                let mut skor = vec![0f32; len];
                let mut q = vec![0f32; d];
                for i in i0..i1 {
                    for ((o, a), b) in q.iter_mut().zip(satir(i, 0)).zip(bq) {
                        *o = (a + b) * olcek;
                    }
                    let mut mx = f32::NEG_INFINITY;
                    for (j, sk) in skor.iter_mut().enumerate() {
                        *sk = nokta(&q, &kh[j * d..][..d]);
                        mx = mx.max(*sk);
                    }
                    let mut top = 0f32;
                    for sk in skor.iter_mut() {
                        *sk = (*sk - mx).exp();
                        top += *sk;
                    }
                    let c = &mut cikti[(i - i0) * d..][..d];
                    for (j, &p) in skor.iter().enumerate() {
                        for (o, &v) in c.iter_mut().zip(&vh[j * d..][..d]) {
                            *o = p.mul_add(v, *o);
                        }
                    }
                    let ters = 1.0 / top;
                    c.iter_mut().for_each(|o| *o *= ters);
                }
                cikti
            })
            .collect();
        let mut y = vec![0f32; self.uzunluk.len() * l * h];
        for (&(b, hd, blk), p) in gorevler.iter().zip(&parcalar) {
            for (n, c) in p.chunks(d).enumerate() {
                y[(b * l + blk * BLOK + n) * h + hd * d..][..d].copy_from_slice(c);
            }
        }
        Ok((
            CpuStorage::F32(y),
            Shape::from_dims(&[self.uzunluk.len() * l, h]),
        ))
    }
}

/// y = x · wᵀ (w f16): küçük m'de `KucukLinear`, aksi halde candle matmul (gemm).
pub fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    if x.dim(0)? <= KUCUK_M {
        x.apply_op2_no_bwd(w, &KucukLinear)
    } else {
        // ponytail: ağırlık her çağrıda f32'ye açılır (katman başı ~50 MB geçici, ~ms); büyük m'de
        // gemm süresinin yanında küçük. Toplu indeksleme bunu ölçülür biçimde yavaş bulursa
        // f16 girdili gemm'e geç.
        x.matmul(&w.apply_op1_no_bwd(&F32ye)?.t()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn rastgele(sekil: &[usize], tohum: u32) -> Tensor {
        let n: usize = sekil.iter().product();
        let mut s = tohum;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        Tensor::from_vec(v, sekil, &Device::Cpu).unwrap()
    }

    fn en_fazla_fark(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap()
    }

    #[test]
    fn kucuk_linear_matmul_ile_ayni() {
        let (x, w) = (rastgele(&[5, 1037], 1), rastgele(&[70, 1037], 2));
        let w16 = w.apply_op1_no_bwd(&F16e).unwrap();
        let a = x.apply_op2_no_bwd(&w16, &KucukLinear).unwrap();
        let b = x
            .matmul(&w16.to_dtype(candle_core::DType::F32).unwrap().t().unwrap())
            .unwrap();
        assert!(en_fazla_fark(&a, &b) < 1e-4);
        // Büyük m yolu (gemm) aynı sonucu verir.
        let x = rastgele(&[KUCUK_M + 3, 1037], 3);
        let a = linear(&x, &w16).unwrap();
        let b = x.apply_op2_no_bwd(&w16, &KucukLinear).unwrap();
        assert!(en_fazla_fark(&a, &b) < 1e-4);
    }

    #[test]
    fn f16_donusumu_candle_ile_ayni() {
        let w = rastgele(&[3, 70_001], 4);
        let a = w.apply_op1_no_bwd(&F16e).unwrap();
        let e = w.to_dtype(candle_core::DType::F16).unwrap();
        assert_eq!(a.to_vec2::<f16>().unwrap(), e.to_vec2::<f16>().unwrap());
        let geri = a.apply_op1_no_bwd(&F32ye).unwrap();
        let e = e.to_dtype(candle_core::DType::F32).unwrap();
        assert_eq!(en_fazla_fark(&geri, &e), 0.0);
    }

    #[test]
    fn bias_gelu_ve_artik_ln_candle_ile_ayni() {
        let x = rastgele(&[7, 96], 3);
        let bias = rastgele(&[96], 4);
        let b = bias.to_vec1::<f32>().unwrap();
        let a = x.apply_op1_no_bwd(&BiasGelu(&b)).unwrap();
        let e = x.broadcast_add(&bias).unwrap().gelu_erf().unwrap();
        assert!(en_fazla_fark(&a, &e) < 1e-6);

        let (r, g, k) = (
            rastgele(&[7, 96], 5),
            rastgele(&[96], 6),
            rastgele(&[96], 7),
        );
        let op = ArtikLn {
            bias: &b,
            agirlik: &g.to_vec1::<f32>().unwrap(),
            kayma: &k.to_vec1::<f32>().unwrap(),
            eps: 1e-5,
        };
        let a = x.apply_op2_no_bwd(&r, &op).unwrap();
        let ln = candle_nn::LayerNorm::new(g, k, 1e-5);
        let e = candle_nn::Module::forward(&ln, &(x.broadcast_add(&bias).unwrap() + &r).unwrap())
            .unwrap();
        assert!(en_fazla_fark(&a, &e) < 1e-5);
    }

    #[test]
    fn dikkat_referans_softmax_ile_ayni_ve_dolgu_etkisiz() {
        let (bas, d, l) = (2usize, 8usize, 5usize);
        let h = bas * d;
        let uz = [5usize, 3];
        let qkv = rastgele(&[2 * l, 3 * h], 9);
        let bias = rastgele(&[3 * h], 10).to_vec1::<f32>().unwrap();
        let op = Dikkat {
            bias: &bias,
            uzunluk: &uz,
            bas,
            bas_boyu: d,
            l,
        };
        let a = qkv.apply_op1_no_bwd(&op).unwrap().to_vec2::<f32>().unwrap();
        // Referans: düz döngüyle, yalnız geçerli anahtarlar.
        let v = qkv.to_vec2::<f32>().unwrap();
        for b in 0..2 {
            for hd in 0..bas {
                for i in 0..uz[b] {
                    let q: Vec<f32> = (0..d)
                        .map(|x| (v[b * l + i][hd * d + x] + bias[hd * d + x]) / (d as f32).sqrt())
                        .collect();
                    let sk: Vec<f32> = (0..uz[b])
                        .map(|j| {
                            (0..d)
                                .map(|x| {
                                    q[x] * (v[b * l + j][h + hd * d + x] + bias[h + hd * d + x])
                                })
                                .sum()
                        })
                        .collect();
                    let mx = sk.iter().cloned().fold(f32::MIN, f32::max);
                    let e: Vec<f32> = sk.iter().map(|s| (s - mx).exp()).collect();
                    let t: f32 = e.iter().sum();
                    for x in 0..d {
                        let bek: f32 = (0..uz[b])
                            .map(|j| {
                                e[j] / t
                                    * (v[b * l + j][2 * h + hd * d + x] + bias[2 * h + hd * d + x])
                            })
                            .sum();
                        assert!((a[b * l + i][hd * d + x] - bek).abs() < 1e-5);
                    }
                }
            }
        }
    }
}
