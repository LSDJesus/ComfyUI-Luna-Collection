# Luna Collection Master Index & Cross-Reference Guide

This document provides a unified index and cross-reference for all technical summaries in the Luna Collection. It lists every module, node, and utility, describes their purpose, and highlights key interactions and dependencies between components. Use this as a quick navigation and integration map for the entire codebase.

---

## **Node Modules**

- **Batch & Loader Nodes**
  - [luna_batch_prompt_extractor](nodes/luna_batch_prompt_extractor.md): Batch prompt extraction; interacts with YAML wildcards and promptcraft engine.
  - [luna_dynamic_loader](nodes/luna_dynamic_loader.md): Dynamic model loading; depends on folder_paths and model_router.
  - [luna_secondary_loader](nodes/luna_secondary_loader.md): Secondary model loader; integrates with model_router and config_gateway.
  - [luna_multi_saver](nodes/luna_multi_saver.md): Multi-output saving; uses upscaling and detailing utilities.

- **Router & API Nodes**
  - [luna_model_router](nodes/luna_model_router.md): Central model routing; interacts with all loader nodes and config_gateway.
  - [luna_daemon_api](nodes/luna_daemon_api.md): Daemon API bridge; connects to luna_daemon server and proxy.
  - [luna_daemon_loader](nodes/luna_daemon_loader.md): Loads models from daemon; depends on daemon API and config.
  - [luna_config_gateway](nodes/luna_config_gateway.md): Configuration management; interacts with loader and router nodes.

- **Prompt & Wildcard Nodes**
  - [luna_expression_pack](nodes/luna_expression_pack.md): Expression parsing; used by promptcraft and wildcard nodes.
  - [luna_yaml_wildcard](nodes/luna_yaml_wildcard.md): YAML-based wildcard system; integrates with logic_engine utility.
  - [luna_wildcard_connections](nodes/luna_wildcard_connections.md): Manages wildcard relationships; depends on yaml_wildcard and promptcraft.

- **LoRA & Validation Nodes**
  - [luna_lora_validator](nodes/luna_lora_validator.md): LoRA metadata validation; interacts with luna_metadata_db utility.

- **Vision & VLM Nodes**
  - [luna_vision_node](nodes/luna_vision_node.md): Vision model integration; uses detailing and upscaling utilities.
  - [luna_vlm_prompt_generator](nodes/luna_vlm_prompt_generator.md): VLM prompt generation; interacts with yaml_wildcard and vision_node.

- **Other Nodes**
  - [luna_gguf_converter](nodes/luna_gguf_converter.md): GGUF model conversion; optional dependency on llama-cpp-python.
  - [luna_zimage_encoder](nodes/luna_zimage_encoder.md): ZImage encoding; used by vision and upscaling nodes.
  - [luna_trigger_injector](nodes/luna_trigger_injector.md): Prompt trigger injection; interacts with batch and loader nodes.
  - [luna_civitai_scraper](nodes/luna_civitai_scraper.md): Civitai metadata scraping; stores results in luna_metadata_db.

- **Promptcraft Submodules**
  - [engine](nodes/promptcraft/engine.md): Prompt template engine; used by batch and wildcard nodes.
  - [nodes](nodes/promptcraft/nodes.md): Promptcraft node definitions.

- **Upscaling Submodules**
  - [luna_super_upscaler](nodes/upscaling/luna_super_upscaler.md): Advanced upscaling; uses tiling and performance_monitor utilities.
  - [luna_ultimate_sd_upscale](nodes/upscaling/luna_ultimate_sd_upscale.md): SD upscaling; integrates with upscaler_simple and advanced.
  - [luna_upscaler_advanced](nodes/upscaling/luna_upscaler_advanced.md): Advanced upscaling logic.
  - [luna_upscaler_simple](nodes/upscaling/luna_upscaler_simple.md): Simple upscaling logic.
  - [seedvr2_wrapper](nodes/upscaling/seedvr2_wrapper.md): SeedVR2 integration.

---

## **Daemon Modules**

- [server](luna_daemon/server.md): Main daemon server; interacts with all CLIP/VAE nodes, LoRA registry, and worker pools.
 - [zimage_proxy](luna_daemon/zimage_proxy.md): Z-IMAGE CLIP proxy; auto-detects Qwen3-based CLIP models and routes encoding to daemon's Qwen3-VL encoder.
- [client](luna_daemon/client.md): Daemon client library; used by daemon_api and loader nodes.
- [config](luna_daemon/config.md): Daemon configuration; referenced by server and loader nodes.
- [proxy](luna_daemon/proxy.md): Proxy classes for VAE/CLIP routing; used by daemon_api and loader nodes.
- [qwen3_encoder](luna_daemon/qwen3_encoder.md): Qwen3-VL encoder; used by vision nodes.
- [README](luna_daemon/README.md): Daemon architecture overview.

---

## **Utility Modules**

- [constants](utils/constants.md): Centralized constants; imported by all nodes and utilities.
- [exceptions](utils/exceptions.md): Custom exception classes; used for error handling in all modules.
- [logic_engine](utils/logic_engine.md): Wildcard resolution engine; used by yaml_wildcard and promptcraft nodes.
- [luna_logger](utils/luna_logger.md): Logging utility; used by all nodes and daemon modules.
- [luna_metadata_db](utils/luna_metadata_db.md): SQLite metadata database; used by lora_validator, civitai_scraper, and loader nodes.
- [luna_performance_monitor](utils/luna_performance_monitor.md): Performance monitoring; used by upscaling and detailing nodes.
- [segs](utils/segs.md): Segmentation utilities; used by vision and detailing nodes.
- [tiling](utils/tiling.md): Tiling orchestration; used by upscaling nodes.
- [trt_engine](utils/trt_engine.md): TensorRT engine wrapper; used by performance nodes (optional).
- [__init__](utils/__init__.md): Central import hub for all utilities.

---

## **Key Interactions & Cross-References**

- **Model Routing:** `luna_model_router` is the central hub, interacting with all loader, config, and daemon nodes.
- **Daemon Integration:** `luna_daemon_api`, `luna_daemon_loader`, and `proxy` connect ComfyUI nodes to the daemon server for shared VAE/CLIP operations.
- **Prompt Generation:** `luna_yaml_wildcard`, `logic_engine`, and `promptcraft/engine` form the backbone of context-aware prompt building.
- **LoRA Metadata:** `luna_lora_validator` and `luna_metadata_db` work together for LoRA validation and metadata caching.
- **Upscaling:** `luna_super_upscaler`, `tiling`, and `performance_monitor` collaborate for efficient high-res image processing.
- **Vision/Detailing:** `segs` and `vision_node` provide segmentation and region-based processing for advanced workflows.
- **Error Handling & Logging:** All nodes and utilities use `exceptions` and `luna_logger` for consistent error reporting and diagnostics.

---

## **How to Use This Guide**
- Click any module name to view its technical summary and API reference.
- Use the "Integration Points" section in each summary for details on dependencies and related modules.
- Refer to this index for a high-level map of how nodes and utilities interact.
- For architectural details, see [luna_daemon/README.md](luna_daemon/README.md) and [server](luna_daemon/luna_daemon_server.py.md).

---

*This index is auto-generated and covers all production modules in the Luna Collection as of December 2025. For updates, regenerate after adding new nodes or utilities.*
