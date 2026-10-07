// Gerçek veri için vektör dosyası: bir klasördeki .md/.txt dosyalarını parçalara böler,
// İbnü'n-Nedîm Gömme'nin (ibnun-nedim) OpenAI uyumlu ucuna (POST /v1/embeddings) gönderir, ham f32 LE yazar.
// Çıktıyı ölçüm alır:
//   bun crates/bge-dizin/deneme/gom.ts --klasor <klasör> [--uc http://127.0.0.1:11434]
//       [--cikti vektorler.f32] [--parca 1200] [--toplu 32]
//   cargo run --release -p bge-dizin --example olc -- gercek vektorler.f32 1024
// .docx okunmaz: önce metne dökün (ör. Word'de "Farklı kaydet → .txt").
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { extname, join } from "node:path";

function arg(ad: string, varsayilan?: string): string | undefined {
  const i = process.argv.indexOf(ad);
  return i > 0 ? process.argv[i + 1] : varsayilan;
}
const klasor = arg("--klasor");
if (!klasor) {
  console.error("kullanım: bun gom.ts --klasor <klasör> [--uc URL] [--cikti dosya] [--parca 1200] [--toplu 32]");
  process.exit(2);
}
const uc = (arg("--uc", "http://127.0.0.1:11434") as string).replace(/\/$/, "");
const cikti = arg("--cikti", "vektorler.f32") as string;
const parcaBoyu = Number(arg("--parca", "1200"));
const toplu = Number(arg("--toplu", "32"));

function dosyalar(d: string): string[] {
  return readdirSync(d).flatMap((ad) => {
    const yol = join(d, ad);
    if (statSync(yol).isDirectory()) return dosyalar(yol);
    return [".md", ".txt"].includes(extname(ad).toLowerCase()) ? [yol] : [];
  });
}

// Paragraf sınırında, en çok `parcaBoyu` karakterlik parçalar (karakter = Unicode kod noktası).
function parcala(metin: string): string[] {
  const parcalar: string[] = [];
  let p = "";
  for (const par of metin.normalize("NFC").split(/\n\s*\n/)) {
    if ([...p].length + [...par].length > parcaBoyu && p) {
      parcalar.push(p.trim());
      p = "";
    }
    p += par + "\n\n";
    while ([...p].length > parcaBoyu) {
      const k = [...p];
      parcalar.push(k.slice(0, parcaBoyu).join("").trim());
      p = k.slice(parcaBoyu).join("");
    }
  }
  if (p.trim()) parcalar.push(p.trim());
  return parcalar.filter((x) => x.length >= 40);
}

const metinler = dosyalar(klasor).flatMap((d) => parcala(readFileSync(d, "utf8")));
if (metinler.length === 0) {
  console.error(`${klasor}: .md/.txt içinde parça çıkmadı`);
  process.exit(1);
}
console.log(`${metinler.length} parça; uç ${uc}`);

const satirlar: Float32Array[] = [];
let boyut = 0;
for (let i = 0; i < metinler.length; i += toplu) {
  const r = await fetch(`${uc}/v1/embeddings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "bge-m3", input: metinler.slice(i, i + toplu) }),
  });
  if (!r.ok) {
    console.error(`HTTP ${r.status}: ${await r.text()}`);
    process.exit(1);
  }
  const j = (await r.json()) as { data: { index: number; embedding: number[] }[] };
  for (const o of j.data.sort((a, b) => a.index - b.index)) {
    if (boyut === 0) boyut = o.embedding.length;
    if (o.embedding.length !== boyut) {
      console.error(`boyut tutarsız: ${o.embedding.length} ≠ ${boyut}`);
      process.exit(1);
    }
    satirlar.push(Float32Array.from(o.embedding));
  }
  process.stdout.write(`\r${satirlar.length}/${metinler.length}`);
}
const tum = new Float32Array(satirlar.length * boyut);
satirlar.forEach((s, i) => tum.set(s, i * boyut));
// Float32Array platformun bayt sırasıyla yazar; x86/ARM'de LE, olc LE okur.
writeFileSync(cikti, new Uint8Array(tum.buffer));
console.log(`\n${cikti}: ${satirlar.length} × ${boyut} f32`);
