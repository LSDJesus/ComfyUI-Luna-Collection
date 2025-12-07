# Luna Collection Master Index & Cross-Reference Guide

This document provides a unified index and cross-reference for all technical summaries in the Luna Collection. It lists every module, node, and utility, describes their purpose, and highlights key interactions and dependencies between components. Use this as a quick navigation and integration map for the entire codebase.

---

## **Node Modules**

  - [luna_batch_prompt_extractor](nodes/luna_batch_prompt_extractor.md): Batch prompt extraction (Luna/Utilities); interacts with YAML wildcards and promptcraft engine.


  - [luna_expression_pack](nodes/luna_expression_pack.md): Expression parsing (Luna); used by promptcraft and wildcard nodes.

  - [luna_lora_validator](nodes/luna_lora_validator.md): LoRA metadata validation (Luna/Utilities); interacts with luna_metadata_db utility.


  - [luna_zimage_encoder](nodes/luna_zimage_encoder.md): ZImage encoding (Luna); used by vision and upscaling nodes.




## **Daemon Modules**

 - [zimage_proxy](luna_daemon/zimage_proxy.md): Z-IMAGE CLIP proxy; auto-detects Qwen3-based CLIP models and routes encoding to daemon's Qwen3-VL encoder.


## **Utility Modules**



## **Key Interactions & Cross-References**



## **How to Use This Guide**


*This index is auto-generated and covers all production modules in the Luna Collection as of December 2025. For updates, regenerate after adding new nodes or utilities.*
