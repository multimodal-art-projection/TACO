<div align="center">

# 🌮 TACO

**A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression**

[![arXiv](https://img.shields.io/badge/arXiv-2604.19572-b31b1b.svg)](http://arxiv.org/abs/2604.19572)
[![HF Daily Paper](https://img.shields.io/badge/%F0%9F%A4%97-Daily%20Paper-yellow)](https://huggingface.co/papers/2604.19572)
[![Code](https://img.shields.io/badge/GitHub-multimodal--art--projection%2FTACO-black?logo=github)](https://github.com/multimodal-art-projection/TACO)

</div>

Terminal agents keep feeding raw shell output back into their own context,
and that noise accumulates quadratically across multi-turn tasks —
drowning out real error signals and inflating token cost.

**TACO** is a plug-and-play, *self-evolving* observational-context
compression framework. Instead of hard-coded truncation, it discovers,
repairs and reuses compression rules online, and keeps a **global rule
pool** that lets new tasks bootstrap from knowledge accumulated on
earlier ones.

On TerminalBench it gives **+1%–4%** across strong backbones
(MiniMax-M2.5, DeepSeek-V3.2, Qwen3-Coder-480B, Qwen3-14B) and transfers
to SWE-Bench Lite, DevEval, CRUST-Bench and CompileBench. See the
[paper](http://arxiv.org/abs/2604.19572) for full results.

---

## Quick start

TACO ships as `terminus-2` inside the Harbor evaluation framework.

```bash
pip install -e .

harbor run \
  -d terminal-bench@2.0 \
  -a terminus-2 \
  -m openai/gpt-4o-mini \
  -n 4 \
  -o results/taco_example \
  --ak enable_compress=True \
  --ak compress_base_url="<COMPRESSION_LLM_URL>" \
  --ak compress_api_key="<COMPRESSION_LLM_KEY>" \
  --ak compress_model_name="<COMPRESSION_LLM_MODEL>" \
  --ak enable_self_evo=True \
  --ak max_turns=200
```

A ready-to-edit template lives at
[`scripts/run_taco_example.sh`](../../../../scripts/run_taco_example.sh).

## Parameters

All flags are passed on the CLI as `--ak <name>=<value>`.

| Flag | Default | Description |
| --- | --- | --- |
| `enable_compress` | `False` | Master switch. Every flag below is a no-op while this is `False`. |
| `compress_base_url` | `""` | Base URL of the self-evo planner LLM (any OpenAI-compatible endpoint). |
| `compress_api_key` | `""` | Bearer token for the planner LLM. |
| `compress_model_name` | `""` | Model name served by the planner LLM. |
| `enable_self_evo` | `False` | Enable the online rule planner / evolver. |
| `freeze_rules` | `False` | Freeze the rule pool — no LLM plan, no evolution (ablation / reproducible runs). |
| `disable_global_evo` | `False` | Ignore the global rule pool and start each task from built-in seed rules (ablation). |
| `uncovered_threshold` | `3000` | Character length above which an *uncovered* output is flagged for new-rule proposal. |
| `max_turns` | `1_000_000` | Per-task turn cap, e.g. `--ak max_turns=200`. |
| `model_info` | `None` | JSON forwarded to LiteLLM, e.g. `'{"max_input_tokens": 132000, "max_output_tokens": 32768}'`. |

## Common configurations

```bash
# Full TACO (recommended)
--ak enable_compress=True --ak enable_self_evo=True \
--ak compress_base_url=... --ak compress_api_key=... --ak compress_model_name=...

# Ablation: freeze the rule pool, no evolution
--ak enable_compress=True --ak enable_self_evo=True --ak freeze_rules=True

# Ablation: local-only evolution (ignore the global pool)
--ak enable_compress=True --ak enable_self_evo=True --ak disable_global_evo=True

# Control (vanilla terminus-2)
--ak enable_compress=False
```

## Citation

```bibtex
@misc{ren2026selfevolvingframeworkefficientterminal,
      title={A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression},
      author={Jincheng Ren and Siwei Wu and Yizhi Li and Kang Zhu and Shu Xu and Boyu Feng and Ruibin Yuan and Wei Zhang and Riza Batista-Navarro and Jian Yang and Chenghua Lin},
      year={2026},
      eprint={2604.19572},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.19572},
}
```
