<script lang="ts">
	import ThemeToggle from '@kleinbem/web-config/theme/toggle';
	import {
		computeEdgeMemory,
		computeParetoIds,
		getMetricX,
		getMetricY,
		listHardwareOptions,
		uniqueQuantsForModel,
		type BenchmarkRecord,
		type SpaceDataset,
		type XMetric,
		type YMetric
	} from '#lib/pareto';

	let dataset = $state.raw<SpaceDataset | null>(null);
	let loadError = $state<string | null>(null);

	let selectedHardware = $state<string>('orin-nano');
	let selectedModel = $state<string>('all');
	let xMetric = $state<XMetric>('mean_kld');
	let yMetric = $state<YMetric>('tg128_tok_s');
	let selectedPointId = $state<string | null>(null);

	// Edge Unified-Memory & KV-Cache Sizer state
	let budgetGib = $state<number>(7.6);
	let contextTokens = $state<number>(8192);
	let kvBits = $state<16 | 8 | 4>(16);
	let sizerModelSlug = $state<string>('qwen3-4b');

	async function loadTelemetry() {
		try {
			const res = await fetch('./data/benchmarks.json');
			if (!res.ok) {
				throw new Error(`HTTP ${res.status}`);
			}
			const data = (await res.json()) as SpaceDataset;
			dataset = data;
			if (data.models.length > 0 && !data.models.some((m) => m.slug === sizerModelSlug)) {
				sizerModelSlug = data.models[0].slug;
			}
			const hwIds = new Set(data.records.map((r) => r.hardware_id));
			if (!hwIds.has(selectedHardware) && data.records.length > 0) {
				selectedHardware = data.records[0].hardware_id;
			}
		} catch (err) {
			loadError = err instanceof Error ? err.message : String(err);
		}
	}

	const initialFetch = loadTelemetry();

	let hardwareOptions = $derived(dataset ? listHardwareOptions(dataset.records) : []);

	let filteredRecords = $derived.by(() => {
		if (!dataset) return [];
		return dataset.records.filter(
			(r) =>
				(selectedHardware === 'all' || r.hardware_id === selectedHardware) &&
				(selectedModel === 'all' || r.model_slug === selectedModel)
		);
	});

	let paretoIds = $derived(computeParetoIds(filteredRecords, xMetric, yMetric));

	const CHART_W = 760;
	const CHART_H = 340;
	const PAD_L = 64;
	const PAD_R = 28;
	const PAD_T = 24;
	const PAD_B = 46;

	let plotted = $derived.by(() => {
		const valid = filteredRecords
			.map((rec) => ({
				rec,
				x: getMetricX(rec, xMetric),
				y: getMetricY(rec, yMetric)
			}))
			.filter(
				(d): d is { rec: BenchmarkRecord; x: number; y: number } =>
					d.rec.status === 'ok' && d.x !== null && d.y !== null
			);

		if (valid.length === 0) {
			return { points: [], linePath: '', xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
		}

		const xs = valid.map((d) => d.x);
		const ys = valid.map((d) => d.y);
		const rawXMin = Math.min(...xs);
		const rawXMax = Math.max(...xs);
		const rawYMin = Math.min(...ys);
		const rawYMax = Math.max(...ys);

		const xSpan = Math.max(1e-6, rawXMax - rawXMin);
		const ySpan = Math.max(1e-6, rawYMax - rawYMin);
		const xMin = Math.max(0, rawXMin - xSpan * 0.08);
		const xMax = rawXMax + xSpan * 0.08;
		const yMin = Math.max(0, rawYMin - ySpan * 0.12);
		const yMax = rawYMax + ySpan * 0.12;

		const innerW = CHART_W - PAD_L - PAD_R;
		const innerH = CHART_H - PAD_T - PAD_B;

		const toSvgX = (v: number) => PAD_L + ((v - xMin) / (xMax - xMin)) * innerW;
		const toSvgY = (v: number) => PAD_T + innerH - ((v - yMin) / (yMax - yMin)) * innerH;

		const points = valid.map((d) => ({
			...d,
			cx: toSvgX(d.x),
			cy: toSvgY(d.y),
			isPareto: paretoIds.has(d.rec.record_id)
		}));

		const frontierPts = points.filter((p) => p.isPareto).sort((a, b) => a.x - b.x);
		const linePath = frontierPts
			.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.cx.toFixed(1)} ${p.cy.toFixed(1)}`)
			.join(' ');

		return { points, linePath, xMin, xMax, yMin, yMax };
	});

	let oomRecords = $derived(filteredRecords.filter((r) => r.status === 'oom'));

	let activePoint = $derived.by(() => {
		if (!selectedPointId) return plotted.points[0] ?? null;
		return plotted.points.find((p) => p.rec.record_id === selectedPointId) ?? plotted.points[0] ?? null;
	});

	let activeSizerArch = $derived(
		dataset?.models.find((m) => m.slug === sizerModelSlug) ?? dataset?.models[0] ?? null
	);

	let sizerRows = $derived.by(() => {
		if (!dataset || !activeSizerArch) return [];
		const isCuda = budgetGib <= 10;
		return uniqueQuantsForModel(dataset.records, activeSizerArch.slug).map((rec) => ({
			rec,
			mem: computeEdgeMemory(
				activeSizerArch,
				rec.weight_gib,
				contextTokens,
				kvBits,
				budgetGib,
				isCuda
			)
		}));
	});

	const xLabels: Record<XMetric, string> = {
		mean_kld: 'Mean KL-Divergence vs FP16 (nats — lower is higher quality)',
		weight_gib: 'Quantized Model Size (GiB — lower fits smaller edge RAM)',
		joules_per_token: 'Energy per Generated Token (Joules/tok — lower is greener)'
	};

	const yLabels: Record<YMetric, string> = {
		tg128_tok_s: 'Decode Speed tg128 (tok/s — higher is faster)',
		pp512_tok_s: 'Prefill Speed pp512 (tok/s — higher is faster)'
	};
</script>

{#snippet metricBadge(fitState: 'comfortable' | 'tight' | 'oom')}
	{#if fitState === 'comfortable'}
		<span class="rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-xs font-medium text-emerald-400">
			Fits Edge RAM
		</span>
	{:else if fitState === 'tight'}
		<span class="rounded-full bg-amber-500/15 px-2.5 py-0.5 text-xs font-medium text-amber-400">
			Tight (&lt;550 MiB)
		</span>
	{:else}
		<span class="rounded-full bg-rose-500/15 px-2.5 py-0.5 text-xs font-medium text-rose-400">
			OOM Cliff
		</span>
	{/if}
{/snippet}

<main class="mx-auto max-w-6xl space-y-8 px-4 py-8 sm:px-6">
	<header class="flex flex-wrap items-start justify-between gap-4 border-b border-white/10 pb-6">
		<div class="space-y-1.5">
			<div class="flex items-center gap-2 text-xs font-semibold tracking-wider uppercase opacity-70">
				<span>kleinbem / hf-bench</span>
				<span>•</span>
				<span>llama.cpp imatrix + CUDA sm_87 Telemetry</span>
			</div>
			<h1 class="text-2xl font-bold tracking-tight sm:text-3xl">
				Edge &amp; Local GGUF Pareto Frontier Explorer
			</h1>
			<p class="max-w-3xl text-sm opacity-80">
				Exact FP16 logit Kullback-Leibler divergence (<code>--kl-divergence</code>), decode throughput,
				unified-memory OOM boundaries, and hardware power efficiency across 8 GiB NVIDIA Jetson Orin
				Nano (<code>sm_87</code> CUDA) and x86_64 Raptor Lake.
			</p>
		</div>
		<div class="flex items-center gap-3">
			<a
				href="https://huggingface.co/datasets/kleinbem/edge-llm-benchmarks"
				target="_blank"
				rel="noreferrer"
				class="surface-card px-3 py-1.5 text-xs font-medium hover:opacity-90"
			>
				Parquet Dataset ↗
			</a>
			<a
				href="https://github.com/kleinbem/hf-bench"
				target="_blank"
				rel="noreferrer"
				class="surface-card px-3 py-1.5 text-xs font-medium hover:opacity-90"
			>
				Nix Flake Source ↗
			</a>
			<ThemeToggle />
		</div>
	</header>

	{#await initialFetch}
		<div class="surface-card p-8 text-sm opacity-75">Loading benchmark telemetry…</div>
	{:then}
		{#if loadError}
			<div class="surface-card border-rose-500/40 p-6 text-sm text-rose-400">
				Failed to load telemetry JSON: {loadError}
			</div>
		{:else if dataset}
			<!-- Section 1: Interactive Pareto Frontier -->
			<section class="surface-card space-y-5 p-5 sm:p-6" aria-labelledby="pareto-heading">
				<div class="flex flex-wrap items-end justify-between gap-4">
					<div>
						<h2 id="pareto-heading" class="text-lg font-semibold">
							1. Quality vs. Speed &amp; Energy Pareto Frontier
						</h2>
						<p class="text-xs opacity-75">
							Connected accent curve highlights non-dominated Pareto-optimal quantizations. Compare
							<code>Q4_K_M</code> (agentic <code>imatrix</code>) against <code>Q4_K_M-std</code> (uncalibrated).
						</p>
					</div>
					<div class="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
						<label class="space-y-1 text-xs">
							<span class="opacity-70">Hardware</span>
							<select
								bind:value={selectedHardware}
								class="w-full rounded-md border border-white/15 bg-transparent px-2.5 py-1.5 text-xs"
							>
								<option value="all">All Hardware</option>
								{#each hardwareOptions as hw (hw.id)}
									<option value={hw.id}>{hw.id}</option>
								{/each}
							</select>
						</label>
						<label class="space-y-1 text-xs">
							<span class="opacity-70">Model</span>
							<select
								bind:value={selectedModel}
								class="w-full rounded-md border border-white/15 bg-transparent px-2.5 py-1.5 text-xs"
							>
								<option value="all">All Models</option>
								{#each dataset.models as m (m.slug)}
									<option value={m.slug}>{m.display_name}</option>
								{/each}
							</select>
						</label>
						<label class="space-y-1 text-xs">
							<span class="opacity-70">X-Axis (Cost/Loss)</span>
							<select
								bind:value={xMetric}
								class="w-full rounded-md border border-white/15 bg-transparent px-2.5 py-1.5 text-xs"
							>
								<option value="mean_kld">Mean KLD (nats)</option>
								<option value="weight_gib">Size (GiB)</option>
								<option value="joules_per_token">Joules / token</option>
							</select>
						</label>
						<label class="space-y-1 text-xs">
							<span class="opacity-70">Y-Axis (Throughput)</span>
							<select
								bind:value={yMetric}
								class="w-full rounded-md border border-white/15 bg-transparent px-2.5 py-1.5 text-xs"
							>
								<option value="tg128_tok_s">Decode tg128 (tok/s)</option>
								<option value="pp512_tok_s">Prefill pp512 (tok/s)</option>
							</select>
						</label>
					</div>
				</div>

				<div class="grid gap-5 lg:grid-cols-[1fr_280px]">
					<div class="overflow-x-auto rounded-lg border border-white/10 p-2">
						{#if plotted.points.length === 0}
							<div class="flex h-64 items-center justify-center text-xs opacity-70">
								No non-OOM measurements have {xMetric} available for this hardware selection. Try
								Mean KLD or Size (GiB).
							</div>
						{:else}
							<svg
								viewBox="0 0 {CHART_W} {CHART_H}"
								class="h-auto w-full min-w-[520px]"
								role="img"
								aria-label="Scatter plot of quantization quality or size versus inference throughput"
							>
								<!-- Axes -->
								<line
									x1={PAD_L}
									y1={CHART_H - PAD_B}
									x2={CHART_W - PAD_R}
									y2={CHART_H - PAD_B}
									stroke="currentColor"
									stroke-opacity="0.25"
								/>
								<line
									x1={PAD_L}
									y1={PAD_T}
									x2={PAD_L}
									y2={CHART_H - PAD_B}
									stroke="currentColor"
									stroke-opacity="0.25"
								/>

								<!-- Axis Bounds Labels -->
								<text x={PAD_L} y={CHART_H - 12} font-size="11" fill="currentColor" opacity="0.7">
									{plotted.xMin.toFixed(xMetric === 'mean_kld' ? 4 : 2)}
								</text>
								<text
									x={CHART_W - PAD_R}
									y={CHART_H - 12}
									text-anchor="end"
									font-size="11"
									fill="currentColor"
									opacity="0.7"
								>
									{plotted.xMax.toFixed(xMetric === 'mean_kld' ? 4 : 2)}
								</text>
								<text
									x={CHART_W / 2}
									y={CHART_H - 10}
									text-anchor="middle"
									font-size="11"
									fill="currentColor"
									opacity="0.85"
								>
									{xLabels[xMetric]}
								</text>
								<text
									x={PAD_L - 8}
									y={PAD_T + 10}
									text-anchor="end"
									font-size="11"
									fill="currentColor"
									opacity="0.7"
								>
									{plotted.yMax.toFixed(1)}
								</text>
								<text
									x={PAD_L - 8}
									y={CHART_H - PAD_B}
									text-anchor="end"
									font-size="11"
									fill="currentColor"
									opacity="0.7"
								>
									{plotted.yMin.toFixed(1)}
								</text>

								<!-- Pareto Frontier Path -->
								{#if plotted.linePath}
									<path
										d={plotted.linePath}
										fill="none"
										stroke="#38bdf8"
										stroke-width="2.25"
										stroke-dasharray="5 3"
									/>
								{/if}

								<!-- Plotted Points -->
								{#each plotted.points as pt (pt.rec.record_id)}
									<g
										class="cursor-pointer"
										role="button"
										tabindex="0"
										aria-label="{pt.rec.model_name} {pt.rec.quant}: {pt.y} tok/s"
										onclick={() => {
											selectedPointId = pt.rec.record_id;
										}}
										onkeydown={(e) => {
											if (e.key === 'Enter' || e.key === ' ') {
												e.preventDefault();
												selectedPointId = pt.rec.record_id;
											}
										}}
									>
										<circle
											cx={pt.cx}
											cy={pt.cy}
											r={pt.isPareto ? 7 : 5.5}
											fill={pt.rec.imatrix_used ? (pt.isPareto ? '#38bdf8' : '#818cf8') : '#f59e0b'}
											stroke="currentColor"
											stroke-width={activePoint?.rec.record_id === pt.rec.record_id ? 2.5 : 1}
										/>
										<text
											x={pt.cx + 9}
											y={pt.cy + 4}
											font-size="11"
											fill="currentColor"
											opacity="0.9"
										>
											{pt.rec.quant}
										</text>
									</g>
								{/each}
							</svg>
						{/if}
					</div>

					<!-- Selected Point Inspector + OOM Cliff Summary -->
					<div class="space-y-4">
						{#if activePoint}
							{@const r = activePoint.rec}
							<div class="rounded-lg border border-white/12 p-4 text-xs space-y-2">
								<div class="flex items-center justify-between gap-2">
									<span class="font-semibold text-sm">{r.model_name} · {r.quant}</span>
									{#if activePoint.isPareto}
										<span class="rounded bg-sky-500/20 px-2 py-0.5 text-[11px] text-sky-300">
											Pareto Optimal
										</span>
									{/if}
								</div>
								<div class="opacity-75">{r.hardware_label}</div>
								<dl class="grid grid-cols-2 gap-y-1.5 pt-2 border-t border-white/10">
									<dt class="opacity-70">Weights / BPW</dt>
									<dd class="text-right font-mono">{r.weight_gib.toFixed(2)} GiB ({r.bpw} bpw)</dd>
									<dt class="opacity-70">Decode tg128</dt>
									<dd class="text-right font-mono">{r.tg128_tok_s?.toFixed(2) ?? '—'} tok/s</dd>
									<dt class="opacity-70">Prefill pp512</dt>
									<dd class="text-right font-mono">{r.pp512_tok_s?.toFixed(1) ?? '—'} tok/s</dd>
									<dt class="opacity-70">Mean KLD</dt>
									<dd class="text-right font-mono">{r.mean_kld?.toFixed(5) ?? '—'} nats</dd>
									<dt class="opacity-70">Top-1 Agreement</dt>
									<dd class="text-right font-mono">
										{r.same_top1_pct !== null ? `${r.same_top1_pct.toFixed(2)}%` : '—'}
									</dd>
									<dt class="opacity-70">Energy / Token</dt>
									<dd class="text-right font-mono">
										{r.joules_per_token !== null ? `${r.joules_per_token.toFixed(3)} J` : '—'}
									</dd>
								</dl>
							</div>
						{/if}

						{#if oomRecords.length > 0}
							<div class="rounded-lg border border-rose-500/30 bg-rose-500/5 p-3.5 text-xs space-y-1.5">
								<div class="font-semibold text-rose-400">
									8 GiB Edge Unified-Memory Cliff ({oomRecords.length} OOM)
								</div>
								{#each oomRecords as oom (oom.record_id)}
									<div class="opacity-85">
										<span class="font-mono font-medium">{oom.model_name} {oom.quant}</span>
										({oom.weight_gib.toFixed(2)} GiB) — exceeds {oom.memory_budget_gib} GiB unified budget.
									</div>
								{/each}
							</div>
						{/if}
					</div>
				</div>
			</section>

			<!-- Section 2: Interactive 8 GiB Edge GQA KV-Cache & Unified-Memory Calculator -->
			<section class="surface-card space-y-5 p-5 sm:p-6" aria-labelledby="sizer-heading">
				<div class="flex flex-wrap items-center justify-between gap-3">
					<div>
						<h2 id="sizer-heading" class="text-lg font-semibold">
							2. Edge Unified-Memory &amp; GQA KV-Cache Sizer
						</h2>
						<p class="text-xs opacity-75">
							Exact memory budget planner accounting for model weights, Grouped-Query Attention KV
							cache, and Tegra Linux + CUDA context overhead.
						</p>
					</div>
					<div class="flex flex-wrap gap-2 text-xs">
						<button
							type="button"
							onclick={() => {
								budgetGib = 7.6;
							}}
							class={[
								'rounded-md border px-2.5 py-1 font-medium',
								budgetGib === 7.6 ? 'border-sky-400 bg-sky-400/15' : 'border-white/15 opacity-80'
							]}
						>
							Jetson Orin Nano (7.6 GiB usable)
						</button>
						<button
							type="button"
							onclick={() => {
								budgetGib = 16;
							}}
							class={[
								'rounded-md border px-2.5 py-1 font-medium',
								budgetGib === 16 ? 'border-sky-400 bg-sky-400/15' : 'border-white/15 opacity-80'
							]}
						>
							16 GiB Edge Node
						</button>
						<button
							type="button"
							onclick={() => {
								budgetGib = 62;
							}}
							class={[
								'rounded-md border px-2.5 py-1 font-medium',
								budgetGib === 62 ? 'border-sky-400 bg-sky-400/15' : 'border-white/15 opacity-80'
							]}
						>
							nixos-nvme (62 GiB)
						</button>
					</div>
				</div>

				<div class="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
					<label class="space-y-1.5 text-xs">
						<span class="font-medium">Architecture</span>
						<select
							bind:value={sizerModelSlug}
							class="w-full rounded-md border border-white/15 bg-transparent px-2.5 py-2 text-xs"
						>
							{#each dataset.models as m (m.slug)}
								<option value={m.slug}>{m.display_name} ({m.num_layers}L / {m.num_kv_heads} KV)</option>
							{/each}
						</select>
					</label>

					<div class="space-y-1 text-xs">
						<label for="budget-slider" class="flex justify-between font-medium">
							<span>RAM / VRAM Budget</span>
							<span class="font-mono">{budgetGib.toFixed(1)} GiB</span>
						</label>
						<input
							id="budget-slider"
							type="range"
							class="edge-slider"
							min="4"
							max="64"
							step="0.2"
							bind:value={budgetGib}
							aria-valuetext="{budgetGib.toFixed(1)} GiB"
							style:--attr-min={4}
							style:--attr-max={64}
							style:--control-value={budgetGib}
						/>
					</div>

					<div class="space-y-1 text-xs">
						<label for="ctx-slider" class="flex justify-between font-medium">
							<span>Context Window</span>
							<span class="font-mono">{contextTokens.toLocaleString()} tok</span>
						</label>
						<input
							id="ctx-slider"
							type="range"
							class="edge-slider"
							min="1024"
							max="32768"
							step="1024"
							bind:value={contextTokens}
							aria-valuetext="{contextTokens} tokens"
							style:--attr-min={1024}
							style:--attr-max={32768}
							style:--control-value={contextTokens}
						/>
					</div>

					<label class="space-y-1.5 text-xs">
						<span class="font-medium">KV-Cache Precision (<code>-ctk/-ctv</code>)</span>
						<select
							bind:value={kvBits}
							class="w-full rounded-md border border-white/15 bg-transparent px-2.5 py-2 text-xs"
						>
							<option value={16}>FP16 (16-bit default)</option>
							<option value={8}>Q8_0 (8-bit KV cache)</option>
							<option value={4}>Q4_0 (4-bit KV cache)</option>
						</select>
					</label>
				</div>

				<div class="space-y-2.5">
					{#each sizerRows as row (row.rec.quant)}
						{@const wPct = Math.min(100, (row.mem.weightsGib / budgetGib) * 100)}
						{@const kvPct = Math.min(100 - wPct, (row.mem.kvCacheGib / budgetGib) * 100)}
						{@const sysPct = Math.min(
							100 - wPct - kvPct,
							(row.mem.runtimeOverheadGib / budgetGib) * 100
						)}
						<div class="rounded-lg border border-white/10 p-3 text-xs space-y-2">
							<div class="flex flex-wrap items-center justify-between gap-2">
								<div class="flex items-center gap-2.5">
									<span class="font-mono font-semibold text-sm">{row.rec.quant}</span>
									<span class="opacity-70">
										Weights {row.mem.weightsGib.toFixed(2)} GiB + KV {row.mem.kvCacheGib.toFixed(2)} GiB
										+ OS/CUDA {row.mem.runtimeOverheadGib.toFixed(2)} GiB =
										<strong>{row.mem.totalGib.toFixed(2)} GiB</strong>
									</span>
								</div>
								<div class="flex items-center gap-2">
									<span class="font-mono opacity-80">
										{row.mem.headroomGib >= 0
											? `+${row.mem.headroomGib.toFixed(2)} GiB free`
											: `${row.mem.headroomGib.toFixed(2)} GiB over`}
									</span>
									{@render metricBadge(row.mem.fitState)}
								</div>
							</div>
							<div class="flex h-2.5 w-full overflow-hidden rounded-full bg-white/10">
								<div
									class="bg-sky-400"
									style:width="{wPct}%"
									title="Weights: {row.mem.weightsGib.toFixed(2)} GiB"
								></div>
								<div
									class="bg-indigo-400"
									style:width="{kvPct}%"
									title="KV Cache: {row.mem.kvCacheGib.toFixed(2)} GiB"
								></div>
								<div
									class="bg-slate-400/70"
									style:width="{sysPct}%"
									title="OS + CUDA Context: {row.mem.runtimeOverheadGib.toFixed(2)} GiB"
								></div>
							</div>
						</div>
					{/each}
				</div>
			</section>

			<!-- Section 3: Responsive Sticky Telemetry Table -->
			<section class="surface-card space-y-4 p-5 sm:p-6" aria-labelledby="table-caption">
				<div
					role="region"
					aria-labelledby="table-caption"
					class="bench-table-wrapper rounded-lg border border-white/10"
				>
					<table class="bench-table text-left text-xs">
						<caption id="table-caption" class="px-4 py-3 text-left text-base font-semibold">
							3. Complete Quantization, KL-Divergence &amp; Hardware Telemetry Matrix
						</caption>
						<thead>
							<tr class="border-b border-white/10 text-[11px] uppercase opacity-75">
								<th class="px-3.5 py-2.5">Model / Quant</th>
								<th class="px-3.5 py-2.5">Hardware</th>
								<th class="px-3.5 py-2.5 text-right">BPW</th>
								<th class="px-3.5 py-2.5 text-right">Size</th>
								<th class="px-3.5 py-2.5 text-right">Mean KLD</th>
								<th class="px-3.5 py-2.5 text-right">Top-1 Match</th>
								<th class="px-3.5 py-2.5 text-right">PPL</th>
								<th class="px-3.5 py-2.5 text-right">Prefill pp512</th>
								<th class="px-3.5 py-2.5 text-right">Decode tg128</th>
								<th class="px-3.5 py-2.5 text-right">Power / J·tok⁻¹</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-white/10">
							{#each filteredRecords as row (row.record_id)}
								<tr class="hover:bg-white/5">
									<th scope="row" class="px-3.5 py-2.5 font-mono font-medium whitespace-nowrap">
										{row.model_name} · {row.quant}
									</th>
									<td data-label="Hardware" class="px-3.5 py-2.5 whitespace-nowrap">
										{row.hardware_id}
									</td>
									<td data-label="BPW" class="px-3.5 py-2.5 text-right font-mono">
										{row.bpw.toFixed(2)}
									</td>
									<td data-label="Size" class="px-3.5 py-2.5 text-right font-mono whitespace-nowrap">
										{row.weight_gib.toFixed(2)} GiB
									</td>
									<td data-label="Mean KLD" class="px-3.5 py-2.5 text-right font-mono">
										{row.mean_kld !== null ? row.mean_kld.toFixed(5) : '—'}
									</td>
									<td data-label="Top-1 Match" class="px-3.5 py-2.5 text-right font-mono">
										{row.same_top1_pct !== null ? `${row.same_top1_pct.toFixed(2)}%` : '—'}
									</td>
									<td data-label="PPL" class="px-3.5 py-2.5 text-right font-mono">
										{row.ppl !== null ? row.ppl.toFixed(3) : '—'}
									</td>
									<td
										data-label="Prefill pp512"
										class="px-3.5 py-2.5 text-right font-mono whitespace-nowrap"
									>
										{row.status === 'oom'
											? 'OOM'
											: row.pp512_tok_s !== null
												? `${row.pp512_tok_s.toFixed(1)} tok/s`
												: '—'}
									</td>
									<td
										data-label="Decode tg128"
										class="px-3.5 py-2.5 text-right font-mono whitespace-nowrap"
									>
										{row.status === 'oom'
											? 'OOM (>8 GiB)'
											: row.tg128_tok_s !== null
												? `${row.tg128_tok_s.toFixed(2)} tok/s`
												: '—'}
									</td>
									<td
										data-label="Power / Energy"
										class="px-3.5 py-2.5 text-right font-mono whitespace-nowrap"
									>
										{#if row.avg_watts !== null && row.joules_per_token !== null}
											{row.avg_watts.toFixed(1)} W ({row.joules_per_token.toFixed(3)} J)
										{:else}
											—
										{/if}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</section>
		{/if}
	{/await}
</main>
