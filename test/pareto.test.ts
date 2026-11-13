import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { computeEdgeMemory, computeParetoIds, type BenchmarkRecord, type ModelArchitecture } from '#lib/pareto';

const qwen3Arch: ModelArchitecture = {
	slug: 'qwen3-4b',
	hf_source: 'Qwen/Qwen3-4B',
	hf_target_repo: 'kleinbem/Qwen3-4B-Edge-GGUF',
	display_name: 'Qwen3-4B',
	params_b: 4.02,
	license_id: 'apache-2.0',
	num_layers: 36,
	num_kv_heads: 8,
	head_dim: 128,
	max_context: 40960
};

function makeRec(id: string, kld: number, tps: number, status: 'ok' | 'oom' = 'ok'): BenchmarkRecord {
	return {
		record_id: id,
		timestamp_utc: '2026-11-13T19:00:00Z',
		model_slug: 'qwen3-4b',
		model_name: 'Qwen3-4B',
		hf_source: 'Qwen/Qwen3-4B',
		hf_target_repo: 'kleinbem/Qwen3-4B-Edge-GGUF',
		params_b: 4.02,
		quant: id,
		imatrix_used: true,
		bpw: 4.85,
		weight_bytes: 2500000000,
		weight_gib: 2.33,
		hardware_id: 'orin-nano',
		hardware_label: 'Jetson Orin Nano',
		backend: 'CUDA',
		memory_budget_gib: 7.6,
		status,
		oom_reason: null,
		pp512_tok_s: 200,
		pp512_std: 1,
		tg128_tok_s: tps,
		tg128_std: 0.2,
		ppl: 6.8,
		ppl_base_fp16: 6.7,
		mean_kld: kld,
		p99_kld: 0.1,
		median_kld: 0.01,
		max_kld: 1.0,
		rms_delta_p_pct: 3.0,
		same_top1_pct: 96.0,
		avg_watts: 11.0,
		joules_per_token: 0.4,
		peak_mem_mib: 3000
	};
}

describe('pareto and edge unified-memory math', () => {
	it('selects strictly non-dominated points and excludes OOM rows', () => {
		const pts = [
			makeRec('Q8_0', 0.002, 18.0),
			makeRec('Q4_K_M', 0.018, 29.5),
			makeRec('Q4_K_M-std', 0.025, 29.1), // dominated by Q4_K_M (higher KLD AND slower)
			makeRec('IQ4_XS', 0.024, 31.2),
			makeRec('F16', 0.0, 0.0, 'oom')
		];
		const frontier = computeParetoIds(pts, 'mean_kld', 'tg128_tok_s');
		assert.deepEqual([...frontier], ['Q8_0', 'Q4_K_M', 'IQ4_XS']);
	});

	it('identifies the 8 GiB unified-memory cliff accurately', () => {
		const fits = computeEdgeMemory(qwen3Arch, 2.33, 8192, 16, 7.6, true);
		assert.equal(fits.fitState, 'comfortable');

		const ooms = computeEdgeMemory(qwen3Arch, 7.55, 8192, 16, 7.6, true);
		assert.equal(ooms.fitState, 'oom');
		assert.ok(ooms.headroomGib < 0);
	});
});
