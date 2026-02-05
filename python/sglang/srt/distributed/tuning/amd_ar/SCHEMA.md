# AMD All-Reduce Tuning JSON Schema

This schema describes the JSON produced by
`benchmark/kernels/all_reduce/benchmark_all_reduce_amd.py --export-tuning`.

## Top-level fields

- `meta`:
  - `device_name`: GPU model string.
  - `world_size`: number of ranks in the group.
  - `dtype`: `"float16" | "bfloat16" | "float32"`.
  - `modes`: list of modes included, e.g. `["eager", "graph"]`.
  - `impls`: list of implementations considered.
  - `size_bytes`: list of explicit sizes in bytes (when provided).
  - `size_powers`: list of size exponents (powers of 2, when used).
  - `min_size_power`, `max_size_power`, `step`: benchmark size range (power-of-2 mode).
  - `respect_impl_constraints`: boolean.
  - `timestamp_utc`: ISO-8601 string.

- `thresholds`:
  - Object keyed by mode, e.g. `{"eager": [...], "graph": [...]}`.
  - Each list element: `{ "max_size_bytes": <int>, "impl": <string> }`.
  - The list is ordered by increasing `max_size_bytes`.
  - For a given mode, the chosen implementation is the fastest for sizes up to
    `max_size_bytes`, then switches to the next entry.

- `guards` (optional):
  - Object keyed by mode, e.g. `{"eager": {...}, "graph": {...}}`.
  - Each mode maps `size_bytes -> { impl: reason }` for guard failures at that size.

## Example

```json
{
  "meta": {
    "device_name": "AMD_Instinct_MI300X",
    "world_size": 8,
    "dtype": "bfloat16",
    "modes": ["graph"],
    "impls": ["torch_native", "pynccl", "custom_ar_sgl", "quick_ar_fp"],
    "size_powers": [15, 16, 17],
    "min_size_power": 15,
    "max_size_power": 17,
    "step": 1,
    "respect_impl_constraints": true,
    "timestamp_utc": "2026-02-04T12:00:00Z"
  },
  "thresholds": {
    "graph": [
      { "max_size_bytes": 65536, "impl": "custom_ar_sgl" },
      { "max_size_bytes": 131072, "impl": "quick_ar_fp" }
    ]
  },
  "guards": {
    "graph": {
      "65536": {
        "custom_ar_aiter": "custom_ar_guard_failed"
      }
    }
  }
}
```
