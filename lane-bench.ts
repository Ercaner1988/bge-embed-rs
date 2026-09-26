// lane-bench.ts - measures the short-query/batch-job lane split (BGE_TOPLU_IZIN)
// under concurrent load: a real search query's latency, and batch throughput,
// while both run against the server at once.
// Usage: BENCH_SOURCE=book.txt bun lane-bench.ts <name=exe[|permit]> [name=exe[|permit] ...]
//   e.g. BENCH_SOURCE=book.txt bun lane-bench.ts gated=./bge-embed-rs.exe|1 ungated=./bge-embed-rs.exe|999
// Env: BENCH_SOURCE (plain-text file, required, >= ~70 KB), BENCH_PORT (11439).
//   --throughput: instead of query latency, measure wall time for a fixed
//   batch workload (throughput cost of the gate), no queries fired.
const THROUGHPUT = Bun.argv.includes("--throughput");
const targets = Bun.argv.slice(2).filter((a) => a !== "--throughput").map((a) => {
  const [name, rest] = a.split("=");
  const [exe, permit] = rest.split("|");
  return { name, exe, permit };
});
if (!process.env.BENCH_SOURCE || targets.length === 0) {
  console.error("usage: BENCH_SOURCE=<text file> bun lane-bench.ts <name=exe[|permit]> [name=exe[|permit] ...] [--throughput]");
  process.exit(1);
}
const PORT = Number(process.env.BENCH_PORT ?? 11439);
const URL = `http://127.0.0.1:${PORT}`;
const source = await Bun.file(process.env.BENCH_SOURCE).text();
const chunk = (i: number) => source.slice(20_000 + i * 1500, 21_500 + i * 1500);

// Representative short search queries (what an interactive lookup sends).
const QUERIES = [
  "root cause analysis method", "search for article", "extract text from a scanned document",
  "weekly review", "convert footnotes to citations", "open a blocked page",
  "read an excel spreadsheet", "extract action items from meeting notes",
];

const embed = (input: string[]) =>
  fetch(`${URL}/v1/embeddings`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model: "bge-m3", input }) })
    .then((r) => { if (!r.ok) throw new Error(`http ${r.status}`); return r.json(); });

async function run(name: string, exe: string, permit?: string) {
  const env: Record<string, string | undefined> = { ...process.env, BGE_PORT: String(PORT), BGE_PARALLEL: "4" };
  if (permit) env.BGE_TOPLU_IZIN = permit; else delete env.BGE_TOPLU_IZIN;
  const proc = Bun.spawn([exe], { env, stdout: "ignore", stderr: "ignore" });
  for (let i = 0; ; i++) {
    try { if ((await fetch(`${URL}/health`)).ok) break; } catch {}
    if (i > 120) throw new Error(`${name} did not come up`);
    await Bun.sleep(1000);
  }
  await embed(["warm-up"]);

  if (THROUGHPUT) {
    // Fixed workload: 3 clients x 2 batch requests x 8 chunks = 48 chunks; measure wall time.
    const t = performance.now();
    await Promise.all([0, 1, 2].map(async (w) => {
      for (let b = 0; b < 2; b++) await embed(Array.from({ length: 8 }, (_, j) => chunk(w * 16 + b * 8 + j)));
    }));
    const seconds = (performance.now() - t) / 1000;
    proc.kill(); await proc.exited; await Bun.sleep(3000);
    return { name, median: 0, worst: 0, throughput: 48 / seconds };
  }

  let stop = false, chunksDone = 0, n = 0;
  const t0 = performance.now();
  const load = [0, 1, 2].map(async () => {
    while (!stop) { const k = (n++ % 40) * 8; await embed(Array.from({ length: 8 }, (_, j) => chunk(k + j))); chunksDone += 8; }
  });
  await Bun.sleep(3000); // let the batch load actually be in flight
  const ms: number[] = [];
  for (const q of QUERIES) { const t = performance.now(); await embed([q]); ms.push(performance.now() - t); }
  const throughput = chunksDone / ((performance.now() - t0) / 1000);
  stop = true; await Promise.all(load); proc.kill(); await proc.exited; await Bun.sleep(3000);
  ms.sort((a, b) => a - b);
  return { name, median: ms[ms.length >> 1], worst: ms.at(-1)!, throughput };
}

const results: Record<string, { median: number; worst: number; throughput: number }[]> = {};
for (const t of targets) results[t.name] = [];
for (let round = 1; round <= 3; round++) {
  const order = targets.map((_, i) => targets[(i + round) % targets.length]); // rotate order each round
  for (const t of order) {
    const r = await run(t.name, t.exe, t.permit);
    results[t.name].push(r);
    console.log(THROUGHPUT
      ? `round ${round} ${t.name}: throughput ${r.throughput.toFixed(2)} chunks/s`
      : `round ${round} ${t.name}: query median ${(r.median / 1000).toFixed(2)} s · worst ${(r.worst / 1000).toFixed(2)} s · throughput ${r.throughput.toFixed(2)} chunks/s`);
  }
}
for (const t of targets) {
  const m = results[t.name].map((r) => r.median / 1000).sort((a, b) => a - b);
  const v = results[t.name].map((r) => r.throughput);
  console.log(`SUMMARY ${t.name}: query medians ${m.map((x) => x.toFixed(2)).join(" / ")} s · mean throughput ${(v.reduce((a, b) => a + b) / v.length).toFixed(2)} chunks/s`);
}
