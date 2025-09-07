import { app } from "/scripts/app.js";

const LunaCheckpointLoaderExtension = {
    name: "luna.checkpointLoader",

    async nodeCreated(node) {
        if (node.type !== "LunaCheckpointLoader") {
            return;
        }

        const previewsWidget = node.widgets.find(w => w.name === "show_previews");

        // --- Create the UI elements ---
        const uiContainer = document.createElement("div");
        const thumbnail = document.createElement("img");
        thumbnail.style.cssText = `width: 100%; margin-bottom: 5px; object-fit: contain;`;
        const metadataDisplay = document.createElement("pre");
        metadataDisplay.style.cssText = `width: 100%; padding: 5px; border: 1px solid #333; border-radius: 4px; background-color: #111; color: #ccc; font-size: 10px; max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-all;`;
        
        uiContainer.appendChild(thumbnail);
        uiContainer.appendChild(metadataDisplay);
        node.root.prepend(uiContainer);

        const checkpointWidget = node.widgets.find(w => w.name === "ckpt_name");

        const getAndDisplayMetadata = async (ckpt_name) => {
            if (!ckpt_name) return;
            try {
                const response = await api.fetchApi(`/luna/get_checkpoint_metadata?ckpt_name=${encodeURIComponent(ckpt_name)}`);
                if (response.status !== 200) throw new Error(`HTTP Error: ${response.status}`);
                const data = await response.json();

                if (data.thumbnail && data.thumbnail.filename) {
                    thumbnail.src = `/view?filename=${encodeURIComponent(data.thumbnail.filename)}&type=${data.thumbnail.type}&subfolder=${encodeURIComponent(data.thumbnail.subfolder || '')}`;
                    thumbnail.style.display = "block";
                } else {
                    thumbnail.style.display = "none";
                }

                const combinedMetadata = { ...data.metadata, ...data.user_lore };
                if (Object.keys(combinedMetadata).length > 0) {
                    metadataDisplay.textContent = JSON.stringify(combinedMetadata, null, 2);
                } else {
                    metadataDisplay.textContent = "No metadata available.";
                }
            } catch (error) {
                console.error(`[Luna] Error fetching metadata for ${ckpt_name}:`, error);
                metadataDisplay.textContent = `Error: ${error.message}`;
            }
        };

        const togglePreviews = (show) => {
            uiContainer.style.display = show ? "block" : "none";
            node.computeSize();
        };

        if (checkpointWidget) {
            const originalCallback = checkpointWidget.callback;
            checkpointWidget.callback = (value) => {
                if (originalCallback) originalCallback.call(node, value);
                getAndDisplayMetadata(value);
            };
        }

        previewsWidget.callback = (value) => {
            togglePreviews(value);
        };
        
        // Initial state calls
        setTimeout(() => {
            getAndDisplayMetadata(checkpointWidget.value);
            togglePreviews(previewsWidget.value);
        }, 10);
    },
};

app.registerExtension(LunaCheckpointLoaderExtension);