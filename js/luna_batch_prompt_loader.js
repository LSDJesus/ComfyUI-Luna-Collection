import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

/**
 * Luna Batch Prompt Loader - JSON file upload support and dynamic index max
 * 
 * Adds file upload functionality and dynamically updates the index max
 * based on the number of entries in the selected JSON file.
 */

app.registerExtension({
    name: "Luna.BatchPromptLoader",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "LunaBatchPromptLoader") {
            return;
        }
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            const jsonFileWidget = this.widgets?.find(w => w.name === "json_file");
            const indexWidget = this.widgets?.find(w => w.name === "index");
            
            if (!jsonFileWidget) {
                return;
            }
            
            // Function to update the index max based on JSON file entry count
            const updateIndexMax = async (filename) => {
                if (!indexWidget) return;
                if (!filename || filename === "No JSON files found" || filename === "Error scanning directory") {
                    return;
                }
                
                try {
                    // Fetch the JSON file to count entries
                    const response = await api.fetchApi(`/view?filename=${encodeURIComponent(filename)}&type=input`);
                    if (response.ok) {
                        const data = await response.json();
                        if (Array.isArray(data) && data.length > 0) {
                            const maxIndex = data.length - 1;
                            indexWidget.options.max = maxIndex;
                            
                            // Clamp current value if it exceeds new max
                            if (indexWidget.value > maxIndex) {
                                indexWidget.value = maxIndex;
                            }
                            
                            console.log(`[Luna] Updated index max to ${maxIndex} (${data.length} entries)`);
                        }
                    }
                } catch (error) {
                    console.log(`[Luna] Could not read JSON file for index max: ${error.message}`);
                }
            };
            
            // Update index max when file selection changes
            const originalCallback = jsonFileWidget.callback;
            jsonFileWidget.callback = async (value) => {
                if (originalCallback) {
                    originalCallback.call(this, value);
                }
                await updateIndexMax(value);
            };
            
            // Initial update for the default selection
            if (jsonFileWidget.value) {
                updateIndexMax(jsonFileWidget.value);
            }
            
            // Create file input element for JSON upload
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = ".json";
            fileInput.style.display = "none";
            document.body.appendChild(fileInput);
            
            // Handle file selection
            fileInput.addEventListener("change", async () => {
                if (fileInput.files.length === 0) {
                    return;
                }
                
                const file = fileInput.files[0];
                
                try {
                    // Upload to ComfyUI input directory
                    const formData = new FormData();
                    formData.append("image", file);  // ComfyUI uses "image" for all uploads
                    formData.append("type", "input");
                    formData.append("overwrite", "true");
                    
                    const response = await api.fetchApi("/upload/image", {
                        method: "POST",
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        const filename = result.name;
                        
                        // Add new file to dropdown options if not already there
                        if (!jsonFileWidget.options.values.includes(filename)) {
                            // Remove placeholder entries if they exist
                            jsonFileWidget.options.values = jsonFileWidget.options.values.filter(
                                v => v !== "No JSON files found" && v !== "Error scanning directory"
                            );
                            jsonFileWidget.options.values.push(filename);
                            jsonFileWidget.options.values.sort();
                        }
                        
                        // Select the uploaded file
                        jsonFileWidget.value = filename;
                        
                        // Update index max for the new file
                        await updateIndexMax(filename);
                        
                        // Force node update
                        if (this.graph) {
                            this.graph.setDirtyCanvas(true);
                        }
                        
                        console.log(`[Luna] Uploaded JSON file: ${filename}`);
                    } else {
                        console.error("[Luna] Failed to upload JSON file:", response.statusText);
                        alert(`Failed to upload file: ${response.statusText}`);
                    }
                } catch (error) {
                    console.error("[Luna] Error uploading JSON file:", error);
                    alert(`Error uploading file: ${error.message}`);
                }
                
                // Reset file input for future uploads
                fileInput.value = "";
            });
            
            // Store reference to file input for cleanup
            this._jsonFileInput = fileInput;
            
            // Add upload button to the node
            const uploadWidget = this.addWidget("button", "upload_json", "Choose JSON File", () => {
                fileInput.click();
            });
            
            // Style the button
            uploadWidget.serialize = false;  // Don't save button state
        };
        
        // Cleanup file input on node removal
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            if (this._jsonFileInput) {
                this._jsonFileInput.remove();
                this._jsonFileInput = null;
            }
            if (onRemoved) {
                onRemoved.apply(this, arguments);
            }
        };
    }
});
