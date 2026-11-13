export interface ModelArchitecture {
	slug: string;
	hf_source: string;
	hf_target_repo: string;
	display_name: string;
	params_b: number;
	license_id: string;
	num_layers: number;
	num_kv_heads: number;
	head_dim: number;
	max_context: number;
}

export interface BenchmarkRecord {
	record_id: string;
	timestamp_utc: string;
	model_slug: string;
	model_name: string;
	hf_source: string;
	hf_target_repo: string;
	params_b: number;
	quant: string;
	imatrix_used: boolean;
	bpw: number;
	weight_bytes: number;
	weight_gib: number;
	hardware_id: string;
	hardware_label: string;
	backend: string;
	memory_budget_gib: number;
	status: 'ok' | 'oom';
	oom_reason: string | null;
	pp512_tok_s: number | null;
	pp512_std: number | null;
	tg128_tok_s: number | null;
	tg128_std: number | null;
	ppl: number | null;
	ppl_base_fp16: number | null;
	mean_kld: number | null;
	p99_kld: number | null;
	median_kld: number | null;
	max_kld: number | null;
	rms_delta_p_pct: number | null;
	same_top1_pct: number | null;
	avg_watts: number | null;
	joules_per_token: number | null;
	peak_mem_mib: number | null;
}

export interface SpaceDataset {
	generatedAt: string;
	sourceRepo: string;
	datasetRepo: string;
	models: ModelArchitecture[];
	records: BenchmarkRecord[];
}

export type XMetric = 'mean_kld' | 'weight_gib' | 'joules_per_token';
export type YMetric = 'tg128_tok_s' | 'pp512_tok_s';

export function getMetricX(rec: BenchmarkRecord, metric: XMetric): number | null {
	const v = rec[metric];
	return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export function getMetricY(rec: BenchmarkRecord, metric: YMetric): number | null {
	const v = rec[metric];
	return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/**
 * Compute non-dominated Pareto-optimal record IDs where lower X (less KLD / smaller GiB / fewer Joules)
 * and higher Y (higher throughput tok/s) strictly dominate.
 */
export function computeParetoIds(
	records: readonly BenchmarkRecord[],
	xMetric: XMetric,
	yMetric: YMetric
): Set<string> {
	const candidates = records
		.map((r) => ({ r, x: getMetricX(r, xMetric), y: getMetricY(r, yMetric) }))
		.filter(
			(item): item is { r: BenchmarkRecord; x: number; y: number } =>
				item.r.status === 'ok' && item.x !== null && item.y !== null && item.y > 0
		)
		.sort((a, b) => a.x - b.x || b.y - a.y);

	const frontier = new Set<string>();
	let maxSeenY = -Infinity;
	for (const item of candidates) {
		if (item.y > maxSeenY) {
			frontier.add(item.r.record_id);
			maxSeenY = item.y;
		}
	}
	return frontier;
}

export interface MemoryBreakdown {
	weightsGib: number;
	kvCacheGib: number;
	runtimeOverheadGib: number;
	totalGib: number;
	headroomGib: number;
	fitState: 'comfortable' | 'tight' | 'oom';
}

export function listHardwareOptions(
	records: readonly BenchmarkRecord[]
): Array<{ id: string; label: string }> {
	const seen = new Map<string, string>();
	for (const r of records) {
		if (!seen.has(r.hardware_id)) {
			seen.set(r.hardware_id, r.hardware_label);
		}
	}
	return Array.from(seen, ([id, label]) => ({ id, label }));
}

export function uniqueQuantsForModel(
	records: readonly BenchmarkRecord[],
	modelSlug: string
): BenchmarkRecord[] {
	const seen = new Map<string, BenchmarkRecord>();
	for (const r of records) {
		if (r.model_slug === modelSlug && !seen.has(r.quant)) {
			seen.set(r.quant, r);
		}
	}
	return Array.from(seen.values()).sort((a, b) => b.bpw - a.bpw);
}

export function computeEdgeMemory(
	arch: ModelArchitecture,
	weightGib: number,
	contextTokens: number,
	kvBits: 16 | 8 | 4,
	budgetGib: number,
	isUnifiedCuda: boolean
): MemoryBreakdown {
	// GQA KV cache: 2 (K+V) * layers * kv_heads * head_dim * bytes_per_elem * contextTokens.
	const kvBytes =
		2 * arch.num_layers * arch.num_kv_heads * arch.head_dim * (kvBits / 8) * contextTokens;
	const kvCacheGib = kvBytes / 1024 ** 3;
	// Unified Tegra RAM shares Linux kernel + page cache (~0.85 GiB) + CUDA/ggml compute buffers (~0.45 GiB).
	const runtimeOverheadGib = isUnifiedCuda ? 1.3 : 0.45;
	const totalGib = weightGib + kvCacheGib + runtimeOverheadGib;
	const headroomGib = budgetGib - totalGib;

	let fitState: MemoryBreakdown['fitState'] = 'comfortable';
	if (headroomGib < 0) {
		fitState = 'oom';
	} else if (headroomGib < 0.55) {
		fitState = 'tight';
	}

	return {
		weightsGib: weightGib,
		kvCacheGib,
		runtimeOverheadGib,
		totalGib,
		headroomGib,
		fitState
	};
}
