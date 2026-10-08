// query-bench.ts - short search-query latency on ONE running server, plus vector agreement with
// a previous run (bench.ts measures the batch path; 3-10 token queries take KucukLinear instead).
// One server at a time on purpose: two bge-m3 copies side by side need ~3.5 GB of RAM.
// Usage: bun query-bench.ts <port> <out.json> [reference.json]
//   e.g. bun query-bench.ts 11435 f32.json   then   bun query-bench.ts 11436 f16.json f32.json
import { readFile, writeFile } from "node:fs/promises";
const [port, out, ref] = process.argv.slice(2);
if (!port || !out) {
  console.error("usage: bun query-bench.ts <port> <out.json> [reference.json]");
  process.exit(1);
}
const queries = [
  "asabiyye", "İbn Haldun devlet kuramı", "Weber karizmatik otorite", "paradigma kayması",
  "hukukun profesyonelleşmesi", "kader ve irade", "nazar ve istidlal", "göçebe toplum dayanışması",
  "bürokratik rasyonalite", "fıkıh usulü kıyas", "medeniyetin çöküşü", "ilm-i umran",
];
async function embed(input: string) {
  const t = performance.now();
  const r = await fetch(`http://127.0.0.1:${port}/v1/embeddings`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "bge-m3", input: [input] }),
  });
  if (!r.ok) throw new Error(`${port} -> HTTP ${r.status}`);
  const j = (await r.json()) as { data: { embedding: number[] }[] };
  return { ms: performance.now() - t, v: j.data[0].embedding };
}
const cos = (a: number[], b: number[]) => { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] ** 2; y += b[i] ** 2; } return d / Math.sqrt(x * y); };
const median = (a: number[]) => { const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };

await embed("ısınma"); // the model may be unloaded while idle
const ms: number[] = [];
const vectors: number[][] = [];
for (let round = 0; round < 3; round++) {
  for (const q of queries) {
    const r = await embed(q);
    ms.push(r.ms);
    if (round === 0) vectors.push(r.v);
  }
}
await writeFile(out, JSON.stringify(vectors));
console.log(`port ${port}: median ${median(ms).toFixed(1)} ms  max ${Math.max(...ms).toFixed(1)} ms  (${ms.length} queries)`);
if (ref) {
  const base = JSON.parse(await readFile(ref, "utf8")) as number[][];
  const c = vectors.map((v, i) => cos(v, base[i]));
  console.log(`cosine vs ${ref}: min ${Math.min(...c).toFixed(6)}  mean ${(c.reduce((a, b) => a + b, 0) / c.length).toFixed(6)}`);
}
