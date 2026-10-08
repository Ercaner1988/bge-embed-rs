//! Süren işler: hangi aracın isteği modeli ya da toplu kapıyı bekliyor, hangisi hesaplanıyor.
//! Connections ekranındaki kuyruk görünümü buradan okur. Bekçi düşünce kayıt silinir. Hesap
//! bekçisi hesabın kendisiyle (spawn_blocking) yaşar: istemci gitse de iş bitene dek görünür
//! ve "sahipsiz" diye işaretlenir (yanıtını bekleyen kalmadı ama CPU'yu hâlâ kullanıyor).

use crate::{Defter, Istek};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evre {
    /// Model bellekte değil; yüklenmesi bekleniyor.
    ModelBekliyor,
    /// Toplu iş, sırası gelsin diye toplu kapıda bekliyor.
    SiraBekliyor,
    /// Arama sorgusu hızlı şeritte hesaplanıyor.
    HizliHesap,
    /// Toplu iş hesaplanıyor.
    TopluHesap,
}

impl Evre {
    pub fn bekliyor_mu(self) -> bool {
        matches!(self, Evre::ModelBekliyor | Evre::SiraBekliyor)
    }
}

#[derive(Clone, Debug)]
pub struct Is {
    pub ad: String,
    pub metin: usize,
    pub evre: Evre,
    pub bas: Instant,
    /// İstemci yanıtı beklemeden gitti; hesap yine de sürüyor.
    pub sahipsiz: bool,
}

pub(crate) struct IsKaydi {
    id: u64,
    istek: u64,
    is: Is,
}

/// Bir evrenin bekçisi; düşünce iş listeden çıkar. `Send`: hesap iş parçacığına taşınabilir.
pub struct IsBekcisi {
    defter: &'static Defter,
    id: u64,
}

impl Drop for IsBekcisi {
    fn drop(&mut self) {
        if let Ok(mut ic) = self.defter.ic.lock() {
            ic.isler.retain(|k| k.id != self.id);
        }
    }
}

impl Istek {
    /// Bu isteğin bir evreye girdiğini yazar (ör. kapıda beklemeye başladı).
    pub fn evre(&self, evre: Evre, metin: usize) -> IsBekcisi {
        let Ok(mut ic) = self.defter.ic.lock() else {
            return IsBekcisi {
                defter: self.defter,
                id: 0,
            };
        };
        ic.sayac += 1;
        let id = ic.sayac;
        ic.isler.push(IsKaydi {
            id,
            istek: self.id,
            is: Is {
                ad: self.ad.clone(),
                metin,
                evre,
                bas: Instant::now(),
                sahipsiz: false,
            },
        });
        IsBekcisi {
            defter: self.defter,
            id,
        }
    }
}

impl Defter {
    /// Süren işler: önce hesaplananlar, sonra bekleyenler; her grupta en eski başta.
    pub fn isler(&self) -> Vec<Is> {
        let Ok(ic) = self.ic.lock() else {
            return Vec::new();
        };
        let mut l: Vec<Is> = ic.isler.iter().map(|k| k.is.clone()).collect();
        l.sort_by_key(|i| (i.evre.bekliyor_mu(), i.bas));
        l
    }

    /// İstek bitmeden düştü: o isteğin süren işleri sahipsiz kaldı.
    pub(crate) fn sahipsiz_birak(&self, istek: u64) {
        if let Ok(mut ic) = self.ic.lock() {
            for k in ic.isler.iter_mut().filter(|k| k.istek == istek) {
                k.is.sahipsiz = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evreler_listelenir_ve_istemci_gidince_sahipsiz_kalir() {
        let d: &'static Defter = Box::leak(Box::default());
        let istek = d.istek("Open Notebook".into());
        let bekle = istek.evre(Evre::SiraBekliyor, 8);
        let hesap_oncesi = d.isler();
        assert_eq!(hesap_oncesi.len(), 1);
        assert!(hesap_oncesi[0].evre.bekliyor_mu());
        drop(bekle);
        let hesap = istek.evre(Evre::TopluHesap, 8);
        drop(istek); // istemci yanıtı beklemeden gitti
        let l = d.isler();
        assert_eq!(
            (l.len(), l[0].evre, l[0].sahipsiz),
            (1, Evre::TopluHesap, true)
        );
        drop(hesap);
        assert!(d.isler().is_empty(), "hesap bitince iş listeden çıkar");
    }

    #[test]
    fn hesaplanan_bekleyenden_once_gelir() {
        let d: &'static Defter = Box::leak(Box::default());
        let a = d.istek("A".into());
        let b = d.istek("B".into());
        let _bekle = a.evre(Evre::ModelBekliyor, 1);
        let _hesap = b.evre(Evre::HizliHesap, 1);
        let l = d.isler();
        assert_eq!((l[0].ad.as_str(), l[1].ad.as_str()), ("B", "A"));
        a.bitir(crate::Sonuc::Tamam);
        b.bitir(crate::Sonuc::Tamam);
    }
}
