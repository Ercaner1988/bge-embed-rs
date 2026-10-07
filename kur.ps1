# İbnü'n-Nedîm Gömme'yi (ibnun-nedim.exe) masaüstündeki mcp-tools klasörüne kurar.
# Yöntem altin-kapi G-20'dekidir: derle, çalışan kopyayı denetle, eski ikiliyi
# yedekle, kopyala, SHA-256'nın kaynakla aynı olduğunu doğrula.
#
# Kullanım (depo kökünden, PowerShell):
#   ./kur.ps1                 # derle ve kur
#   ./kur.ps1 -Durdur         # hedefteki ikili çalışıyorsa önce durdur
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

if (-not $DerlemeYok) {
    Push-Location $Kok
    try {
        cargo build --release --bin ibnun-nedim
        if ($LASTEXITCODE -ne 0) { throw "derleme başarısız (çıkış $LASTEXITCODE)" }
    } finally { Pop-Location }
}

New-Item -ItemType Directory -Force -Path $Hedef | Out-Null
$zaman = Get-Date -Format 'yyyyMMdd-HHmmss'
foreach ($ad in $Ikililer) {
    $kaynak = Join-Path $Kok "target\release\$ad.exe"
    $hedefYol = Join-Path $Hedef "$ad.exe"
    if (-not (Test-Path $kaynak)) { throw "$kaynak yok; -DerlemeYok verildiyse önce derleyin" }

    $calisan = @(Get-Process -Name $ad -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $hedefYol })
    if ($calisan.Count -gt 0) {
        if (-not $Durdur) {
            throw "$hedefYol çalışıyor (PID $($calisan.Id -join ', ')). Kapatın ya da -Durdur ile yeniden koşun."
        }
        $calisan | Stop-Process -Force
        $calisan | Wait-Process -Timeout 10 -ErrorAction SilentlyContinue
        Write-Host "durduruldu: $ad (PID $($calisan.Id -join ', '))"
    }

    $yeni = (Get-FileHash -Algorithm SHA256 $kaynak).Hash
    if (Test-Path $hedefYol) {
        if ((Get-FileHash -Algorithm SHA256 $hedefYol).Hash -eq $yeni) {
            Write-Host "güncel: $hedefYol"
            continue
        }
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
Write-Host "Oturum açılışında başlatma açıksa, bu kopyayı bir kez açın: kayıt kendini bu yola günceller."
