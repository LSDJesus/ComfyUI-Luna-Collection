import { app } from "/scripts/app.js";

// This script adds dynamic UI controls to our Pyrite Core nodes.

app.registerExtension({
    name: "PyriteCore.AdvancedUpscalerUI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // We are looking for our specific node, the Pyrite Advanced Upscaler.
        if (nodeData.name === "Pyrite_AdvancedUpscaler") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Find the specific widgets we need to control.
                const strategyWidget = this.widgets.find((w) => w.name === "tile_strategy");
                const modeWidget = this.widgets.find((w) => w.name === "tile_mode");
                const resolutionWidget = this.widgets.find((w) => w.name === "tile_resolution");
                const overlapWidget = this.widgets.find((w) => w.name === "tile_overlap");

                // Store their original properties so we can restore them.
                if (modeWidget) modeWidget.origType = modeWidget.type;
                if (resolutionWidget) resolutionWidget.origType = resolutionWidget.type;
                if (overlapWidget) overlapWidget.origType = overlapWidget.type;

                // This function will be called whenever the strategy dropdown changes.
                const updateVisibility = () => {
                    const strategy = strategyWidget.value;
                    
                    if (strategy === "none") {
                        // If strategy is 'none', hide the other tiling widgets.
                        if (modeWidget) modeWidget.type = "hidden";
                        if (resolutionWidget) resolutionWidget.type = "hidden";
                        if (overlapWidget) overlapWidget.type = "hidden";
                    } else {
                        // Otherwise, show them by restoring their original type.
                        if (modeWidget) modeWidget.type = modeWidget.origType;
                        if (resolutionWidget) resolutionWidget.type = resolutionWidget.origType;
                        if (overlapWidget) overlapWidget.type = overlapWidget.origType;
                    }
                    // Refresh the node's display.
                    this.computeSize();
                };

                // Add our function as a callback to the strategy widget.
                strategyWidget.callback = updateVisibility;
                
                // Run it once right now to set the initial state.
                updateVisibility();
            };
        }
    },
});