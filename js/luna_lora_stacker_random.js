import { app } from "/scripts/app.js";

// This is the main extension object for our Random LoRA Stacker.
const LunaLoRAStackerRandomExtension = {
    name: "luna.loraStackerRandom",

    // This MUST match the Python file's MAX_LORA_SLOTS.
    MAX_LORA_SLOTS: 4,

    async nodeCreated(node) {
        // We bind our logic only to our specific Random Stacker node.
        if (node.type !== "LunaLoRAStackerRandom") {
            return;
        }

        // --- Find the master toggle for showing/hiding the UI ---
        const previewsWidget = node.widgets.find(w => w.name === "show_previews");
        // We'll store all our created UI containers here for easy access.
        const uiContainers = [];

        // --- Reusable function to fetch and display metadata ---
        // This is identical to our other Oracles, a testament to our modular design.
        const getAndDisplayLoraMetadata = async (lora_name, thumbnailElement, metadataElement) => {
            if (!lora_name || lora_name === "None") {
                thumbnailElement.style.display = "none";
                metadataElement.style.display = "none";
                return;
            }
            try {
                const response = await api.fetchApi(`/luna/get_lora_metadata?lora_name=${encodeURIComponent(lora_name)}`);
                if (response.status !== 200) throw new Error(`HTTP Error: ${response.status}`);
                const data = await response.json();

                if (data.thumbnail && data.thumbnail.filename) {
                    thumbnailElement.src = `/view?filename=${encodeURIComponent(data.thumbnail.filename)}&type=${data.thumbnail.type}&subfolder=${encodeURIComponent(data.thumbnail.subfolder || '')}`;
                    thumbnailElement.style.display = "block";
                } else {
                    thumbnailElement.style.display = "none";
                }

                const combinedMetadata = { ...data.metadata, ...data.user_lore };
                if (Object.keys(combinedMetadata).length > 0) {
                    metadataElement.textContent = JSON.stringify(combinedMetadata, null, 2);
                    metadataElement.style.display = "block";
                } else {
                    metadataElement.textContent = "No metadata available.";
                    metadataElement.style.display = "block";
                }
            } catch (error) {
                console.error(`[Luna] Error fetching metadata for ${lora_name}:`, error);
                metadataElement.textContent = `Error: ${error.message}`;
                metadataElement.style.display = "block";
            }
        };

        // --- Master function to toggle all previews on/off ---
        const togglePreviews = (show) => {
            for (const container of uiContainers) {
                container.style.display = show ? "block" : "none";
            }
            node.computeSize();
        };

        // --- Loop to build the UI for each LoRA slot ---
        for (let i = 1; i <= this.MAX_LORA_SLOTS; i++) {
            const loraNameWidget = node.widgets.find(w => w.name === `lora_name_${i}`);
            if (loraNameWidget) {
                const uiContainer = document.createElement("div");
                uiContainers.push(uiContainer);

                const thumbnail = document.createElement("img");
                thumbnail.style.cssText = `width: 100%; margin-bottom: 5px; object-fit: contain;`;
                uiContainer.appendChild(thumbnail);

                const metadataDisplay = document.createElement("pre");
                metadataDisplay.style.cssText = `width: 100%; padding: 5px; border: 1px solid #333; border-radius: 4px; background-color: #111; color: #ccc; font-size: 10px; max-height: 150px; overflow-y: auto; white-space: pre-wrap; word-break: break-all;`;
                uiContainer.appendChild(metadataDisplay);
                
                // For the Random Stacker, the last widget in each slot is the precision control.
                // We will insert our UI container right after it to keep everything grouped.
                const lastWidgetInSlot = node.widgets.find(w => w.name === `precision_model_${i}`);
                if (lastWidgetInSlot.inputEl) {
                     lastWidgetInSlot.inputEl.parentElement.after(uiContainer);
                } else { // Fallback
                     loraNameWidget.inputEl.parentElement.after(uiContainer);
                }

                // Hijack the callback for the LoRA dropdown in this slot.
                const originalCallback = loraNameWidget.callback;
                loraNameWidget.callback = (value) => {
                    if (originalCallback) originalCallback.call(node, value);
                    getAndDisplayLoraMetadata(value, thumbnail, metadataDisplay);
                };
                
                // Initial call to populate the UI for this slot.
                setTimeout(() => getAndDisplayLoraMetadata(loraNameWidget.value, thumbnail, metadataDisplay), 10);
            }
        }
        
        // --- Attach the logic for our master show/hide toggle ---
        previewsWidget.callback = (value) => {
            togglePreviews(value);
        };

        // --- Set the initial state for the entire node's UI ---
        setTimeout(() => togglePreviews(previewsWidget.value), 20);
    },
};

// Register our extension with the ComfyUI app.
app.registerExtension(LunaLoRAStackerRandomExtension);