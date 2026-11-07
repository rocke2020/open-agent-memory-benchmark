# BEAM dataset attribution and license

The published questions, gold answers, rubrics and source-derived contexts originate from BEAM by Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani and J Ross Mitchell, described in [Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs](https://arxiv.org/abs/2510.27246).

Source: [Mohammadta/BEAM-10M](https://huggingface.co/datasets/Mohammadta/BEAM-10M), dataset revision `9b2096193fe74e2837e4713e483351e19817773c`. The upstream [dataset license declaration](https://huggingface.co/datasets/Mohammadta/BEAM-10M/blob/main/README.md) is [Creative Commons Attribution–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).

The compressed per-question data in this directory are shared under CC BY-SA 4.0. Changes include OAM retrieval/context selection, model-generated answers and judgments, and aggregation of recorded first-pass and retry outcomes; the original questions, gold answers and rubrics are retained. These records do not constitute a redistribution of the full conversation dataset.

Saved prompts include the AMB/Mem0-origin nugget-judge template. Its separate Apache-2.0 attribution and license are retained in [prompt provenance](../../../prompt-packs/README.md#native-oam-campaign-prompts) and [LICENSE.Mem0](../../../prompt-packs/LICENSE.Mem0). These notices do not change the repository code license.
