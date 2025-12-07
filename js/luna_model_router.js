/**
 * Luna Model Router - Frontend JavaScript
 * 
 * Handles:
 * 1. Dynamic filtering of model dropdown based on model_source selection
 * 2. CLIP slot label updates based on model_type
 * 3. Visual hints for required vs optional CLIP slots
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Cache for model lists by source
const modelListCache = {
    checkpoints: null,
    diffusion_models: null,
    unet: null,
};

// CLIP requirements by model type (mirrors Python CLIP_REQUIREMENTS)
const CLIP_REQUIREMENTS = {
    "SD1.5": {
        required: ["clip_1"],
        labels: {
            clip_1: "clip_1 (CLIP-L) *",
            clip_2: "clip_2 (not used)",
            clip_3: "clip_3 (not used)",
            clip_4: "clip_4 (not used)",
        }
    },
    "SDXL": {
        required: ["clip_1", "clip_2"],
        labels: {
            clip_1: "clip_1 (CLIP-L) *",
            clip_2: "clip_2 (CLIP-G) *",
            clip_3: "clip_3 (not used)",
            clip_4: "clip_4 (not used)",
        }
    },
    "SDXL + Vision": {
        required: ["clip_1", "clip_2", "clip_4"],
        labels: {
            clip_1: "clip_1 (CLIP-L) *",
            clip_2: "clip_2 (CLIP-G) *",
            clip_3: "clip_3 (not used)",
            clip_4: "clip_4 (Vision) *",
        }
    },
    "Flux": {
        required: ["clip_1", "clip_3"],
        labels: {
            clip_1: "clip_1 (CLIP-L) *",
            clip_2: "clip_2 (not used)",
            clip_3: "clip_3 (T5-XXL) *",
            clip_4: "clip_4 (not used)",
        }
    },
    "Flux + Vision": {
        required: ["clip_1", "clip_3", "clip_4"],
        labels: {
            clip_1: "clip_1 (CLIP-L) *",
            clip_2: "clip_2 (not used)",
            clip_3: "clip_3 (T5-XXL) *",
            clip_4: "clip_4 (Vision) *",
        }
    },
    "SD3": {
        required: ["clip_1", "clip_2", "clip_3"],
        labels: {
            clip_1: "clip_1 (CLIP-L) *",
            clip_2: "clip_2 (CLIP-G) *",
            clip_3: "clip_3 (T5-XXL) *",
            clip_4: "clip_4 (not used)",
        }
    },
    "Z-IMAGE": {
        required: ["clip_1"],
        labels: {
            clip_1: "clip_1 (Qwen3) *",
            clip_2: "clip_2 (not used)",
            clip_3: "clip_3 (not used)",
            clip_4: "clip_4 (not used)",
        }
    },
};

// Fetch model list for a specific source
async function fetchModelList(source) {
    if (modelListCache[source] !== null) {
        return modelListCache[source];
    }
    
    try {
        // Use our custom API endpoint
        const response = await api.fetchApi(`/luna/model_list/${source}`);
        if (response.ok) {
            const data = await response.json();
            modelListCache[source] = data.models || [];
        } else {
            modelListCache[source] = [];
        }
    } catch (e) {
        console.warn(`[LunaModelRouter] Failed to fetch ${source} list:`, e);
        modelListCache[source] = [];
    }
    
    return modelListCache[source];
}

// Update model_name dropdown based on model_source
async function updateModelDropdown(node, source) {
    const modelNameWidget = node.widgets?.find(w => w.name === "model_name");
    if (!modelNameWidget) return;
    
    // Get models for this source
    const models = await fetchModelList(source);
    
    // Update widget options
    const options = ["None", ...models];
    modelNameWidget.options.values = options;
    
    // Reset to None if current value not in new list
    if (!options.includes(modelNameWidget.value)) {
        modelNameWidget.value = "None";
    }
    
    // Trigger redraw
    node.setDirtyCanvas(true);
}

// Update CLIP widget labels based on model_type
function updateClipLabels(node, modelType) {
    const requirements = CLIP_REQUIREMENTS[modelType];
    if (!requirements) return;
    
    for (let i = 1; i <= 4; i++) {
        const widgetName = `clip_${i}`;
        const widget = node.widgets?.find(w => w.name === widgetName);
        if (widget) {
            // Update the widget's displayed name
            widget.label = requirements.labels[widgetName];
            
            // Visual indicator: make required slots more prominent
            if (requirements.required.includes(widgetName)) {
                widget.inputEl?.classList?.add("luna-clip-required");
            } else {
                widget.inputEl?.classList?.remove("luna-clip-required");
            }
        }
    }
    
    node.setDirtyCanvas(true);
}

// Register the extension
app.registerExtension({
    name: "Luna.ModelRouter",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "LunaModelRouter") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            
            const node = this;
            
            // === Handle model_source changes ===
            const sourceWidget = node.widgets?.find(w => w.name === "model_source");
            if (sourceWidget) {
                const originalCallback = sourceWidget.callback;
                
                sourceWidget.callback = async function(value) {
                    if (originalCallback) {
                        originalCallback.call(this, value);
                    }
                    await updateModelDropdown(node, value);
                };
                
                // Initialize with current value
                setTimeout(() => {
                    updateModelDropdown(node, sourceWidget.value || "checkpoints");
                }, 100);
            }
            
            // === Handle model_type changes ===
            const typeWidget = node.widgets?.find(w => w.name === "model_type");
            if (typeWidget) {
                const originalCallback = typeWidget.callback;
                
                typeWidget.callback = function(value) {
                    if (originalCallback) {
                        originalCallback.call(this, value);
                    }
                    updateClipLabels(node, value);
                };
                
                // Initialize with current value
                setTimeout(() => {
                    updateClipLabels(node, typeWidget.value || "SDXL");
                }, 100);
            }
            
            return result;
        };
    },
    
    async setup() {
        // Pre-fetch model lists on startup
        for (const source of Object.keys(modelListCache)) {
            fetchModelList(source);
        }
        
        // Add CSS for required CLIP slots
        const style = document.createElement("style");
        style.textContent = `
            .luna-clip-required {
                border-left: 3px solid #4a90d9 !important;
            }
        `;
        document.head.appendChild(style);
    }
});
