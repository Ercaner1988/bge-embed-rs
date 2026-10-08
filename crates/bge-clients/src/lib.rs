//! Bu sunucuya istek gönderen yerel araçların defteri (Connections ekranı, "Local tools").
//!
//! İstemciler kendini tanıtmaz: uzun ömürlü bağlantı yok, User-Agent çoğunda ureq'in
//! varsayılanı. Windows'ta isteği taşıyan TCP bağlantısının sahibi süreç bulunur ve exe
//! adı bilinen araçlarla eşlenir. Defter süreç geneli tektir (`defter()`): sunucu yazar,
//! pencere okur.
//!
//! "Bağlantı koptu" sunucunun gözünden üç biçimde görünür: istek hata döndü, istemci yanıtı
//! beklemeden ayrıldı (zaman aşımı; ör. Nazar 2 sn bekler, model yüklenirken yetişmez) ya
//! da gömme sırasında panik. Sunucunun kendisi kapalıyken bu pencere de kapalıdır.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

/// Exe adında geçen parça → görünen ad (ilk eşleşen kazanır).
const ADA_GORE: &[(&str, &str)] = &[
    ("nazard", "Nazar"),
    ("nazar-mcp", "Nazar"),
    ("ibnunnedim", "el-Fihrist"),
    ("pencere-egui", "Paslı Beyin"),
    ("pasli", "Paslı Beyin"),
    ("kesfuzzunun", "Keşfü'z-Zunûn"),
    ("zopay", "Keşfü'z-Zunûn"),
    ("graphify", "graphify"),
    // llm-for-zotero eklentisi Zotero sürecinin içinde koşar.
    ("zotero", "Zotero (llm-for-zotero)"),
    ("wslrelay", "Docker"),
    ("com.docker", "Docker"),
    ("vpnkit", "Docker"),
];
/// Exe yolunda geçen parça → görünen ad: Python'la koşan araçlar exe adıyla ayrılmaz.
const YOLA_GORE: &[(&str, &str)] = &[
    ("open-notebook", "Open Notebook"),
    ("open_notebook", "Open Notebook"),
];
/// İstek gelmese de listede durur: hiç istek gelmemesi de bilinmesi gereken bir durumdur.
pub const BEKLENEN: &[&str] = &[
    "Nazar",
    "el-Fihrist",
    "Paslı Beyin",
    "Keşfü'z-Zunûn",
    "Open Notebook",
];
const OLAY_SINIRI: usize = 30;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sonuc {
    Tamam,
    Hata(String),
    /// İstemci yanıtı beklemeden bağlantıyı kapattı (kendi zaman aşımı).
    Ayrildi,
    Panik(String),
}

#[derive(Clone, Debug)]
pub struct Sorun {
    pub zaman: SystemTime,
    pub metin: String,
    pub panik: bool,
    /// Sorundan sonra aynı araçtan başarılı istek geldi.
    pub duzeldi: bool,
}

#[derive(Clone, Debug)]
pub struct Kayit {
    pub ad: String,
    pub istek: u64,
    pub hatali: u64,
    pub son: Option<SystemTime>,
    pub son_sure_ms: u64,
    pub son_sorun: Option<Sorun>,
}

impl Kayit {
    fn yeni(ad: &str) -> Self {
        Self {
            ad: ad.to_string(),
            istek: 0,
            hatali: 0,
            son: None,
            son_sure_ms: 0,
            son_sorun: None,
        }
    }

    /// Çözülmemiş sorun varsa o.
    pub fn acik_sorun(&self) -> Option<&Sorun> {
        self.son_sorun.as_ref().filter(|s| !s.duzeldi)
    }
}

#[derive(Clone, Debug)]
pub struct Olay {
    pub zaman: SystemTime,
    pub metin: String,
    /// Kırmızı gösterilir (hata, panik); değilse bilgi (model atıldı/yüklendi).
    pub agir: bool,
}

#[derive(Default)]
pub struct Defter {
    ic: Mutex<Ic>,
}

#[derive(Default)]
struct Ic {
    kayitlar: Vec<Kayit>,
    olaylar: VecDeque<Olay>,
}

/// Süreç geneli defter.
pub fn defter() -> &'static Defter {
    static D: OnceLock<Defter> = OnceLock::new();
    D.get_or_init(Defter::default)
}

impl Defter {
    pub fn kaydet(&self, ad: &str, sure: Duration, sonuc: Sonuc) {
        let Ok(mut ic) = self.ic.lock() else { return };
        let simdi = SystemTime::now();
        let k = match ic.kayitlar.iter().position(|k| k.ad == ad) {
            Some(i) => &mut ic.kayitlar[i],
            None => {
                ic.kayitlar.push(Kayit::yeni(ad));
                let son = ic.kayitlar.len() - 1;
                &mut ic.kayitlar[son]
            }
        };
        k.istek += 1;
        k.son = Some(simdi);
        k.son_sure_ms = sure.as_millis() as u64;
        let ms = k.son_sure_ms;
        let (metin, panik) = match sonuc {
            Sonuc::Tamam => {
                if let Some(s) = k.son_sorun.as_mut() {
                    s.duzeldi = true;
                }
                return;
            }
            Sonuc::Hata(e) => (format!("{ad}: request failed after {ms} ms: {e}"), false),
            Sonuc::Ayrildi => (
                format!(
                    "{ad} gave up after {ms} ms and closed the connection before the answer \
                     (the model may have been loading, or the server was busy)"
                ),
                false,
            ),
            Sonuc::Panik(e) => (format!("{ad}: embedding panicked: {e}"), true),
        };
        k.hatali += 1;
        k.son_sorun = Some(Sorun {
            zaman: simdi,
            metin: metin.clone(),
            panik,
            duzeldi: false,
        });
        olay_ekle(&mut ic.olaylar, metin, true);
    }

    /// Bir araca bağlı olmayan sunucu olayı (model atıldı/yüklendi, yükleme hatası).
    pub fn olay(&self, metin: impl Into<String>, agir: bool) {
        if let Ok(mut ic) = self.ic.lock() {
            olay_ekle(&mut ic.olaylar, metin.into(), agir);
        }
    }

    /// Beklenen araçlar (istek gelmemişse boş kayıtla) + görülen öbürleri; olaylar yeniden eskiye.
    pub fn anlik(&self) -> (Vec<Kayit>, Vec<Olay>) {
        let Ok(ic) = self.ic.lock() else {
            return (Vec::new(), Vec::new());
        };
        let mut l: Vec<Kayit> = BEKLENEN
            .iter()
            .map(|ad| {
                ic.kayitlar
                    .iter()
                    .find(|k| k.ad == *ad)
                    .cloned()
                    .unwrap_or_else(|| Kayit::yeni(ad))
            })
            .collect();
        l.extend(
            ic.kayitlar
                .iter()
                .filter(|k| !BEKLENEN.contains(&k.ad.as_str()))
                .cloned(),
        );
        (l, ic.olaylar.iter().rev().cloned().collect())
    }

    /// İstek başında çağrılır; `bitir` çağrılmadan düşerse istemci ayrılmış sayılır.
    pub fn istek(&'static self, ad: String) -> Istek {
        Istek {
            defter: self,
            ad,
            bas: Instant::now(),
            bitti: false,
        }
    }
}

fn olay_ekle(l: &mut VecDeque<Olay>, metin: String, agir: bool) {
    if l.len() == OLAY_SINIRI {
        l.pop_front();
    }
    l.push_back(Olay {
        zaman: SystemTime::now(),
        metin,
        agir,
    });
}

/// Süren bir isteğin bekçisi. axum, istemci bağlantıyı kapatınca işleyiciyi düşürür; bekçi de
/// düşer ve `bitir` görmediği için bunu "istemci ayrıldı" diye yazar.
pub struct Istek {
    defter: &'static Defter,
    ad: String,
    bas: Instant,
    bitti: bool,
}

impl Istek {
    pub fn bitir(mut self, sonuc: Sonuc) {
        self.bitti = true;
        self.defter.kaydet(&self.ad, self.bas.elapsed(), sonuc);
    }
}

impl Drop for Istek {
    fn drop(&mut self) {
        if !self.bitti {
            self.defter
                .kaydet(&self.ad, self.bas.elapsed(), Sonuc::Ayrildi);
        }
    }
}

/// Exe yolundan görünen ad; bilinmiyorsa exe adı.
pub fn ad_bul(exe_yolu: &str) -> String {
    let yol = exe_yolu.replace('\\', "/").to_ascii_lowercase();
    let dosya = yol.rsplit('/').next().unwrap_or(&yol);
    if let Some((_, ad)) = ADA_GORE.iter().find(|(p, _)| dosya.contains(p)) {
        return (*ad).to_string();
    }
    if let Some((_, ad)) = YOLA_GORE.iter().find(|(p, _)| yol.contains(p)) {
        return (*ad).to_string();
    }
    let ozgun = exe_yolu.rsplit(['\\', '/']).next().unwrap_or(exe_yolu);
    ozgun.strip_suffix(".exe").unwrap_or(ozgun).to_string()
}

/// İsteği gönderen aracın adı. Bulunamazsa uzak adres (ör. ağdan gelen istek).
pub fn tanimla(uzak: SocketAddr, sunucu_portu: u16) -> String {
    if !uzak.ip().is_loopback() {
        return format!("network: {}", uzak.ip());
    }
    match sahip_exe(uzak, sunucu_portu) {
        Some(yol) => ad_bul(&yol),
        None => "unknown local process".to_string(),
    }
}

#[cfg(windows)]
fn sahip_exe(uzak: SocketAddr, sunucu_portu: u16) -> Option<String> {
    surec_yolu(sahip_pid(uzak, sunucu_portu)?)
}

#[cfg(not(windows))]
fn sahip_exe(_: SocketAddr, _: u16) -> Option<String> {
    None
}

/// İstemcinin bu bağlantı için açtığı soketin sahibi: yerel portu isteğin kaynak portu,
/// uzak portu bu sunucunun portu olan IPv4 satırı (sunucunun kendi satırı tersidir).
#[cfg(windows)]
fn sahip_pid(uzak: SocketAddr, sunucu_portu: u16) -> Option<u32> {
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetExtendedTcpTable, MIB_TCPROW_OWNER_PID, MIB_TCPTABLE_OWNER_PID,
        TCP_TABLE_OWNER_PID_CONNECTIONS,
    };
    use windows_sys::Win32::Networking::WinSock::AF_INET;
    let SocketAddr::V4(v4) = uzak else {
        return None;
    };
    let mut boyut = 0u32;
    // SAFETY: boş tamponla yalnız gereken boyut sorulur.
    unsafe {
        GetExtendedTcpTable(
            std::ptr::null_mut(),
            &mut boyut,
            0,
            u32::from(AF_INET),
            TCP_TABLE_OWNER_PID_CONNECTIONS,
            0,
        )
    };
    // u32 hizalı tampon; iki çağrı arasında açılan bağlantılar için pay.
    let mut tampon = vec![0u32; (boyut as usize).div_ceil(4) + 256];
    let mut boyut = (tampon.len() * 4) as u32;
    // SAFETY: tampon `boyut` bayt yazılabilir ve tablo yapısının hizasında.
    let r = unsafe {
        GetExtendedTcpTable(
            tampon.as_mut_ptr().cast(),
            &mut boyut,
            0,
            u32::from(AF_INET),
            TCP_TABLE_OWNER_PID_CONNECTIONS,
            0,
        )
    };
    if r != 0 {
        return None;
    }
    let tablo = tampon.as_ptr().cast::<MIB_TCPTABLE_OWNER_PID>();
    // SAFETY: başarılı çağrı tabloyu doldurdu; satırlar `table`dan başlayıp `dwNumEntries` tane.
    let satirlar: &[MIB_TCPROW_OWNER_PID] = unsafe {
        let n = (*tablo).dwNumEntries as usize;
        std::slice::from_raw_parts(std::ptr::addr_of!((*tablo).table).cast(), n)
    };
    let port = |p: u32| u16::from_be(p as u16);
    satirlar
        .iter()
        .find(|s| port(s.dwLocalPort) == v4.port() && port(s.dwRemotePort) == sunucu_portu)
        .map(|s| s.dwOwningPid)
}

#[cfg(windows)]
fn surec_yolu(pid: u32) -> Option<String> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
    };
    // SAFETY: tutamak yalnız bu çerçevede kullanılıp kapatılır; tampon boyu `n`le verilir.
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return None;
        }
        let mut tampon = [0u16; 1024];
        let mut n = tampon.len() as u32;
        let ok = QueryFullProcessImageNameW(h, 0, tampon.as_mut_ptr(), &mut n);
        CloseHandle(h);
        (ok != 0).then(|| String::from_utf16_lossy(&tampon[..n as usize]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exe_yolundan_arac_adi() {
        assert_eq!(
            ad_bul(r"C:\Users\x\AppData\Local\nazar\bin\nazard.exe"),
            "Nazar"
        );
        assert_eq!(
            ad_bul(r"C:\x\mcp-tools\el-fihrist\ibnunnedim.exe"),
            "el-Fihrist"
        );
        assert_eq!(
            ad_bul(r"C:\x\.cargo\bin\kesfuzzunun-baglayici.exe"),
            "Keşfü'z-Zunûn"
        );
        assert_eq!(
            ad_bul(r"C:\x\open-notebook\.venv\Scripts\python.exe"),
            "Open Notebook"
        );
        assert_eq!(ad_bul(r"C:\Python\python.exe"), "python");
    }

    #[test]
    fn ayrilan_istemci_ve_duzelme() {
        let d: &'static Defter = Box::leak(Box::default());
        drop(d.istek("Nazar".into()));
        let (l, olaylar) = d.anlik();
        let n = l.iter().find(|k| k.ad == "Nazar").unwrap();
        assert!(n.acik_sorun().unwrap().metin.contains("gave up"));
        assert_eq!((n.istek, n.hatali, olaylar.len()), (1, 1, 1));
        d.istek("Nazar".into()).bitir(Sonuc::Tamam);
        let (l, _) = d.anlik();
        let n = l.iter().find(|k| k.ad == "Nazar").unwrap();
        assert!(
            n.acik_sorun().is_none(),
            "sonraki başarılı istek sorunu kapatır"
        );
        assert_eq!(l.len(), BEKLENEN.len(), "beklenenler hep listede");
    }

    #[test]
    fn olaylar_sinirli_ve_yeniden_eskiye() {
        let d = Defter::default();
        for i in 0..OLAY_SINIRI + 5 {
            d.olay(format!("o{i}"), false);
        }
        let (_, o) = d.anlik();
        assert_eq!(o.len(), OLAY_SINIRI);
        assert_eq!(o[0].metin, format!("o{}", OLAY_SINIRI + 4));
    }

    /// Gerçek bir yerel TCP bağlantısının sahibi bu süreç olarak bulunur.
    #[cfg(windows)]
    #[test]
    fn baglanti_sahibi_bu_surec() {
        let dinle = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = dinle.local_addr().unwrap().port();
        let istemci = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        let (_kabul, uzak) = dinle.accept().unwrap();
        assert_eq!(uzak.port(), istemci.local_addr().unwrap().port());
        assert_eq!(sahip_pid(uzak, port), Some(std::process::id()));
        let ben = std::env::current_exe().unwrap().display().to_string();
        assert_eq!(
            sahip_exe(uzak, port).map(|y| y.to_ascii_lowercase()),
            Some(ben.to_ascii_lowercase())
        );
    }
}
