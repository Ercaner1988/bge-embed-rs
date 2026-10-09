# İbnü'n-Nedîm Gömme'yi (ibnun-nedim.exe) masaüstündeki mcp-tools klasörüne kurar.
# Yöntem altin-kapi G-20'dekidir: derle, çalışan kopyayı denetle, eski ikiliyi
# yedekle, kopyala, SHA-256'nın kaynakla aynı olduğunu doğrula.
#
# Kullanım (depo kökünden, PowerShell):
#   ./kur.ps1                 # derle ve kur
#   ./kur.ps1 -Durdur         # ikili değiştiyse ve çalışıyorsa önce durdur; 11434 hizmetini sonra yeniden açar
#   ./kur.ps1 -DerlemeYok     # target\release'teki hazır ikiliyi kur
#   ./kur.ps1 -Hedef D:\baska # başka klasöre
param(
    [string]$Hedef = (Join-Path $env:USERPROFILE 'Desktop\mcp-tools\ibnun-nedim'),
    [switch]$Durdur,
    [switch]$DerlemeYok
)
$ErrorActionPreference = 'Stop'
$Ikililer = @('ibnun-nedim')
$Kok = $PSScriptRoot
# Oturumun gömme hizmeti: ~/.llama-embed-server/start.ps1 onu 11434'te açar (Nazar, el-Fihrist, Open Notebook).
$HizmetPortu = 11434
$HizmetBetigi = Join-Path $env:USERPROFILE '.llama-embed-server\start.ps1'
$hizmetiAc = $false

if (-not $DerlemeYok) {
    Push-Location $Kok
    try {
        cargo build --release --bin ibnun-nedim
        if ($LASTEXITCODE -ne 0) { throw "derleme başarısız (çıkış $LASTEXITCODE)" }
    } finally { Pop-Location }
}

New-Item -ItemType Directory -Force -Path $Hedef | Out-Null
$zaman = Get-Date -Format 'yyyyMMdd-HHmmss'
# start.ps1 yalnız varsayılan kopyayı açar; başka -Hedef'teki hizmet yeniden açılamaz, uyarılır.
$varsayilanHedef = $Hedef -eq (Join-Path $env:USERPROFILE 'Desktop\mcp-tools\ibnun-nedim')
# Kurulum ortada düşse de (Move/Copy/SHA throw) durdurulan hizmet yeniden açılır: finally.
try {
foreach ($ad in $Ikililer) {
    $kaynak = Join-Path $Kok "target\release\$ad.exe"
    $hedefYol = Join-Path $Hedef "$ad.exe"
    if (-not (Test-Path $kaynak)) { throw "$kaynak yok; -DerlemeYok verildiyse önce derleyin" }

    # Önce karşılaştır: ikili güncelse çalışan hizmete dokunulmaz. (2026-10-10: durdurma
    # karşılaştırmadan önceydi; güncel ikilide de 11434 hizmeti ölüyor, Nazar/el-Fihrist
    # anlam kanalı yeniden başlatılana dek BM25'e düşüyordu.)
    $yeni = (Get-FileHash -Algorithm SHA256 $kaynak).Hash
    if ((Test-Path $hedefYol) -and (Get-FileHash -Algorithm SHA256 $hedefYol).Hash -eq $yeni) {
        Write-Host "güncel: $hedefYol"
        continue
    }

    $calisan = @(Get-Process -Name $ad -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $hedefYol })
    if ($calisan.Count -gt 0) {
        if (-not $Durdur) {
            throw "$hedefYol çalışıyor (PID $($calisan.Id -join ', ')). Kapatın ya da -Durdur ile yeniden koşun."
        }
        # Hizmet (start.ps1'in açtığı, 11434) kurulumdan sonra yeniden açılır; pencereler açılmaz.
        $dinleyen = $calisan | ForEach-Object {
            Get-NetTCPConnection -OwningProcess $_.Id -State Listen -ErrorAction SilentlyContinue
        } | Where-Object LocalPort -eq $HizmetPortu
        if ($dinleyen) { $hizmetiAc = $true }
        $calisan | Stop-Process -Force
        $calisan | Wait-Process -Timeout 10 -ErrorAction SilentlyContinue
        Write-Host "durduruldu: $ad (PID $($calisan.Id -join ', '))"
    }

    if (Test-Path $hedefYol) {
        $yedek = Join-Path $Hedef "$ad.$zaman.eski.exe"
        Move-Item $hedefYol $yedek
        Write-Host "yedek: $yedek"
    }
    Copy-Item $kaynak $hedefYol
    if ((Get-FileHash -Algorithm SHA256 $hedefYol).Hash -ne $yeni) {
        throw "$hedefYol kopyası kaynakla aynı değil (SHA-256)"
    }
    Write-Host "kuruldu: $hedefYol (SHA-256 $($yeni.Substring(0, 16)))"
}
} finally {
    if ($hizmetiAc) {
        if (-not $varsayilanHedef) {
            Write-Warning "hizmet ($HizmetPortu) durduruldu; start.ps1 yalnız varsayılan kopyayı açar, $Hedef için elle başlatın"
        } elseif (-not (Test-Path $HizmetBetigi)) {
            Write-Warning "hizmet ($HizmetPortu) durduruldu ama $HizmetBetigi yok: elle başlatın"
        } else {
            & powershell -NoProfile -ExecutionPolicy Bypass -File $HizmetBetigi
            # start.ps1 port doluysa sessizce çıkar: başlatıldığını süreçten doğrula.
            $acik = Get-Process -Name 'ibnun-nedim' -ErrorAction SilentlyContinue |
                Where-Object { $_.Path -eq (Join-Path $Hedef 'ibnun-nedim.exe') }
            if ($acik) {
                Write-Host "hizmet yeniden başlatıldı (PID $($acik.Id -join ', ')); model yüklenince $HizmetPortu dinlenir"
            } else {
                Write-Warning "start.ps1 çağrıldı ama ibnun-nedim süreci yok: $HizmetBetigi'yi elle çalıştırın"
            }
        }
    }
}
Write-Host "Oturum açılışında başlatma açıksa, bu kopyayı bir kez açın: kayıt kendini bu yola günceller."
