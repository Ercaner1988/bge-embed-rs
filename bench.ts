// bench.ts - compares embedding backends on identical input, rotating the order every round.
// Usage: BENCH_SOURCE=book.txt bun bench.ts <name=port> [name=port ...]
//   e.g. BENCH_SOURCE=book.txt bun bench.ts native=11434 v3=11435
// Env: BENCH_SOURCE (plain-text file, required, >= ~70 KB), BENCH_ROUNDS (3), BENCH_BATCH (8 chunks/request).
const targets = Bun.argv.slice(2).map((a) => { const [name, port] = a.split("="); return { name, url: `http://127.0.0.1:${port}/v1/embeddings` }; });
const ROUNDS = Number(process.env.BENCH_ROUNDS ?? 3);
const BATCH = Number(process.env.BENCH_BATCH ?? 8);
if (!process.env.BENCH_SOURCE || targets.length === 0) {
  console.error("usage: BENCH_SOURCE=<text file> bun bench.ts <name=port> [name=port ...]");
  process.exit(1);
}
const source = await Bun.file(process.env.BENCH_SOURCE).text();

// ~1500-char chunks (a typical RAG chunk size); every round uses fresh chunks.
const chunks: string[] = [];
for (let i = 20_000; chunks.length < BATCH * (ROUNDS + 1); i += 1500) chunks.push(source.slice(i, i + 1500));

async function embed(url: string, input: string[]): Promise<{ s: number; v: number[][] }> {
  const t = performance.now();
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: "bge-m3", input }) });
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const j = (await r.json()) as { data: { embedding: number[]; index: number }[] };
  return { s: (performance.now() - t) / 1000, v: j.data.sort((a, b) => a.index - b.index).map((d) => d.embedding) };
}

const cos = (a: number[], b: number[]) => { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] ** 2; y += b[i] ** 2; } return d / Math.sqrt(x * y); };
const median = (a: number[]) => { const s = [...a].sort((p, q) => p - q); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

// Warm-up + agreement: the first batch once per backend, vectors compared pairwise.
const warmup = chunks.slice(0, BATCH);
const ref: Record<string, number[][]> = {};
for (const t of targets) ref[t.name] = (await embed(t.url, warmup)).v;

const timings: Record<string, number[]> = Object.fromEntries(targets.map((t) => [t.name, []]));
for (let r = 0; r < ROUNDS; r++) {
  const batch = chunks.slice(BATCH * (r + 1), BATCH * (r + 2));
  const order = targets.map((_, i) => targets[(i + r) % targets.length]); // rotate order each round
  for (const t of order) {
    const { s } = await embed(t.url, batch);
    timings[t.name].push(s / batch.length);
    console.log(`round ${r + 1} ${t.name.padEnd(8)} ${(s / batch.length).toFixed(2)} s/chunk`);
  }
}

console.log("\n== Result (seconds per chunk, lower is better) ==");
for (const t of targets) {
  const a = timings[t.name];
  console.log(`${t.name.padEnd(8)} median ${median(a).toFixed(2)}  min ${Math.min(...a).toFixed(2)}  max ${Math.max(...a).toFixed(2)}`);
}
console.log("\n== Vector agreement (mean cosine over the warm-up batch) ==");
for (let i = 0; i < targets.length; i++) for (let k = i + 1; k < targets.length; k++) {
  const a = ref[targets[i].name], b = ref[targets[k].name];
  const mean = a.reduce((s, v, j) => s + cos(v, b[j]), 0) / a.length;
  console.log(`${targets[i].name} ~ ${targets[k].name}: ${mean.toFixed(6)}`);
}
