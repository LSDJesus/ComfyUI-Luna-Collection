import { app } from "/scripts/app.js";

// The extension is now named for its true purpose.
const LunaLoRAStackerExtension = {
    // The extension's internal name is now purified.
    name: "luna.loraStacker",

    // This MUST match the Python file's MAX_LORA_SLOTS.
    MAX_LORA_SLOTS: 4,

    async nodeCreated(node) {
        // The check now looks for the true, final name of the node.
        if (node.type !== "LunaLoRAStacker") {
            return;
        }

        const previewsWidget = node.widgets.find(w => w.name === "show_previews");
        const uiContainers = []; // We will store our UI containers here to toggle them all at once.

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

        const togglePreviews = (show) => {
            for (const container of uiContainers) {
                container.style.display = show ? "block" : "none";
            }
            node.computeSize();
        };

        // --- Loop to build UI for each slot ---
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
                
                const clipStrengthWidget = node.widgets.find(w => w.name === `clip_strength_${i}`);
                if (clipStrengthWidget.inputEl) {
                     clipStrengthWidget.inputEl.parentElement.after(uiContainer);
                } else {
                     loraNameWidget.inputEl.parentElement.after(uiContainer);
                }

                const originalCallback = loraNameWidget.callback;
                loraNameWidget.callback = (value) => {
                    if (originalCallback) originalCallback.call(node, value);
                    getAndDisplayLoraMetadata(value, thumbnail, metadataDisplay);
                };
                
                setTimeout(() => getAndDisplayLoraMetadata(loraNameWidget.value, thumbnail, metadataDisplay), 10);
            }
        }
        
        // --- Attach the master toggle's logic ---
        previewsWidget.callback = (value) => {
            togglePreviews(value);
        };

        // --- Set the initial state ---
        setTimeout(() => togglePreviews(previewsWidget.value), 20);
    },
};

// Register our perfected extension with the ComfyUI app.
app.registerExtension(LunaLoRAStackerExtension);