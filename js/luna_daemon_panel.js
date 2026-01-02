import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

/**
 * Luna Daemon Panel
 * 
 * A sidebar panel for managing the Luna VAE/CLIP Daemon.
 * Shows status, allows configuration of max workers and TTL.
 */

// Panel state
let panelState = {
    running: false,
    error: null,
    device: "unknown",
    vramUsed: 0,
    vramAllocated: 0,
    vramTotal: 0,
    vramPercent: 0,
    requests: 0,
    clipRequests: 0,
    vaeRequests: 0,
    uptime: 0,
    modelsLoaded: [],
    checkpoints: [],
    gpus: [],
    vaeLoaded: false,
    clipLoaded: false,
    weightRegistryModels: [],
    comfyuiVram: null,
};

// Create the panel element
function createPanelContent() {
    const container = document.createElement("div");
    container.className = "luna-daemon-panel";
    container.innerHTML = `
        <style>
            .luna-daemon-panel {
                padding: 10px;
                font-family: system-ui, -apple-system, sans-serif;
                font-size: 13px;
                color: var(--fg-color);
            }
            .luna-daemon-panel h3 {
                margin: 0 0 12px 0;
                font-size: 14px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .luna-daemon-panel .status-indicator {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                display: inline-block;
            }
            .luna-daemon-panel .status-indicator.running {
                background: #4ade80;
                box-shadow: 0 0 6px #4ade80;
            }
            .luna-daemon-panel .status-indicator.stopped {
                background: #f87171;
                box-shadow: 0 0 6px #f87171;
            }
            .luna-daemon-panel .error-message {
                color: #f87171;
                font-size: 11px;
                margin-top: 4px;
                padding: 4px;
                background: rgba(248, 113, 113, 0.1);
                border-radius: 4px;
                word-break: break-word;
            }
            .luna-daemon-panel .section {
                background: var(--comfy-input-bg);
                border-radius: 6px;
                padding: 10px;
                margin-bottom: 10px;
            }
            .luna-daemon-panel .section-title {
                font-weight: 500;
                margin-bottom: 8px;
                font-size: 12px;
                text-transform: uppercase;
                opacity: 0.7;
            }
            .luna-daemon-panel .stat-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 4px;
            }
            .luna-daemon-panel .stat-label {
                opacity: 0.8;
            }
            .luna-daemon-panel .stat-value {
                font-weight: 500;
            }
            .luna-daemon-panel .vram-bar {
                height: 6px;
                background: var(--border-color);
                border-radius: 3px;
                margin-top: 6px;
                overflow: hidden;
            }
            .luna-daemon-panel .vram-fill {
                height: 100%;
                background: linear-gradient(90deg, #4ade80, #fbbf24);
                border-radius: 3px;
                transition: width 0.3s ease;
            }
            .luna-daemon-panel .models-list {
                font-size: 12px;
                opacity: 0.9;
            }
            .luna-daemon-panel .model-item {
                padding: 4px 0;
                border-bottom: 1px solid var(--border-color);
            }
            .luna-daemon-panel .model-item:last-child {
                border-bottom: none;
            }
            .luna-daemon-panel .setting-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            .luna-daemon-panel .setting-input {
                width: 70px;
                padding: 4px 8px;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                background: var(--comfy-input-bg);
                color: var(--fg-color);
                font-size: 12px;
            }
            .luna-daemon-panel .btn {
                padding: 6px 12px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                font-weight: 500;
                transition: opacity 0.2s;
            }
            .luna-daemon-panel .btn:hover {
                opacity: 0.8;
            }
            .luna-daemon-panel .btn-unload {
                background: #fbbf24;
                color: #000;
            }
            .luna-daemon-panel .btn-start {
                background: #4ade80;
                color: #000;
            }
            .luna-daemon-panel .btn-stop {
                background: #f87171;
                color: #fff;
            }
            .luna-daemon-panel .btn-refresh {
                background: var(--border-color);
                color: var(--fg-color);
            }
            .luna-daemon-panel .button-row {
                display: flex;
                gap: 8px;
                margin-top: 12px;
            }
            .luna-daemon-panel .no-daemon {
                text-align: center;
                padding: 20px;
                opacity: 0.6;
            }
        </style>
        
        <h3>
            <span class="status-indicator ${panelState.running ? 'running' : 'stopped'}"></span>
            Luna Daemon
        </h3>
        
        ${panelState.error ? `<div class="error-message">${panelState.error}</div>` : ''}
        
        <div class="section">
            <div class="section-title">Status</div>
            <div class="stat-row">
                <span class="stat-label">State</span>
                <span class="stat-value" id="luna-status">${panelState.running ? 'Running (System Tray)' : 'Stopped'}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Device</span>
                <span class="stat-value" id="luna-device">${panelState.device}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Requests (Total)</span>
                <span class="stat-value" id="luna-requests">${panelState.requests}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">CLIP Requests</span>
                <span class="stat-value" id="luna-clip-requests">${panelState.clipRequests}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">VAE Requests</span>
                <span class="stat-value" id="luna-vae-requests">${panelState.vaeRequests}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Uptime</span>
                <span class="stat-value" id="luna-uptime">${formatUptime(panelState.uptime)}</span>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">VRAM - Multi-GPU</div>
            ${panelState.gpus.length > 0 
                ? (() => {
                    console.log(`[Luna Panel] Building UI for ${panelState.gpus.length} GPUs:`, panelState.gpus);
                    return panelState.gpus.map(gpu => `
                        <div style="margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color);">
                            <div class="stat-row">
                                <span class="stat-label">GPU ${gpu.id}${gpu.is_daemon_device ? ' (Daemon)' : ''}</span>
                                <span class="stat-value">${gpu.reserved_gb} / ${gpu.total_gb} GB</span>
                            </div>
                            <div class="stat-row" style="font-size: 11px; opacity: 0.7;">
                                <span>${gpu.name}</span>
                                <span>${gpu.percent}%</span>
                            </div>
                            <div class="vram-bar">
                                <div class="vram-fill" style="width: ${gpu.percent}%; ${gpu.percent > 90 ? 'background: #f87171;' : ''}"></div>
                            </div>
                        </div>
                    `).join('');
                })()
                
                : `
                <div class="stat-row">
                    <span class="stat-label">Reserved</span>
                    <span class="stat-value" id="luna-vram">${panelState.vramUsed.toFixed(2)} / ${panelState.vramTotal.toFixed(1)} GB</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Active</span>
                    <span class="stat-value" id="luna-vram-allocated">${panelState.vramAllocated.toFixed(2)} GB</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Usage</span>
                    <span class="stat-value" id="luna-vram-percent">${panelState.vramPercent.toFixed(1)}%</span>
                </div>
                <div class="vram-bar">
                    <div class="vram-fill" id="luna-vram-bar" style="width: ${panelState.vramPercent}%"></div>
                </div>
                `
            }
        </div>
        
        ${panelState.comfyuiVram ? `
        <div class="section">
            <div class="section-title">ComfyUI Instance VRAM</div>
            <div class="stat-row">
                <span class="stat-label">Device</span>
                <span class="stat-value">${panelState.comfyuiVram.device}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Used</span>
                <span class="stat-value">${panelState.comfyuiVram.used_gb} / ${panelState.comfyuiVram.total_gb} GB</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Usage</span>
                <span class="stat-value">${panelState.comfyuiVram.percent}%</span>
            </div>
            <div class="vram-bar">
                <div class="vram-fill" style="width: ${panelState.comfyuiVram.percent}%; ${panelState.comfyuiVram.percent > 90 ? 'background: #f87171;' : ''}"></div>
            </div>
            ${panelState.comfyuiVram.loaded_models_count !== undefined ? `
            <div class="stat-row" style="margin-top: 8px;">
                <span class="stat-label">Loaded Models</span>
                <span class="stat-value">${panelState.comfyuiVram.loaded_models_count}</span>
            </div>
            ` : ''}
        </div>
        ` : ''}
        
        ${panelState.weightRegistryModels && panelState.weightRegistryModels.length > 0 ? `
        <div class="section">
            <div class="section-title">Daemon Weight Registry</div>
            <div class="models-list" style="max-height: 300px; overflow-y: auto;">
                ${panelState.weightRegistryModels.map(model => `
                    <div class="model-item" style="margin-bottom: 8px; padding: 8px; background: rgba(255,255,255,0.02); border-radius: 4px;">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div style="flex: 1; min-width: 0;">
                                <div style="font-weight: 500; color: #4ade80;">${model.type.toUpperCase()}</div>
                                <div style="font-size: 11px; opacity: 0.7; margin-top: 2px; word-break: break-all;">
                                    ${model.path || model.clip_l_path || model.key}
                                </div>
                            </div>
                            <div style="text-align: right; margin-left: 8px; white-space: nowrap;">
                                <div style="font-weight: 600; color: #fbbf24;">${model.memory_gb} GB</div>
                                <div style="font-size: 10px; opacity: 0.6; margin-top: 2px;">${model.tensor_count} tensors</div>
                            </div>
                        </div>
                        <div style="font-size: 10px; opacity: 0.5; margin-top: 4px;">
                            Device: ${model.device}
                        </div>
                    </div>
                `).join('')}
            </div>
            <div style="margin-top: 8px; padding: 6px; background: rgba(251, 191, 36, 0.1); border-radius: 4px; font-size: 11px;">
                <strong>Total:</strong> ${panelState.weightRegistryModels.reduce((sum, m) => sum + m.memory_gb, 0).toFixed(2)} GB across ${panelState.weightRegistryModels.length} model(s)
            </div>
        </div>
        ` : ''}
        
        <div class="section">
            <div class="section-title">Device Configuration</div>
            <div id="luna-device-config">
                <div style="text-align: center; padding: 12px; opacity: 0.6;">
                    Loading devices...
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Daemon Models</div>
            <div class="models-list" id="luna-models">
                ${panelState.modelsLoaded.length > 0 
                    ? panelState.modelsLoaded.map(m => `<div class="model-item">${m}</div>`).join('')
                    : '<div style="opacity: 0.5">Waiting for first workflow...</div>'
                }
            </div>
            ${panelState.modelsLoaded.length > 0 
                ? '<button class="btn btn-unload" id="luna-unload" style="margin-top: 8px; width: 100%;">Unload Models</button>'
                : ''
            }
        </div>
        
        <div class="section">
            <div class="section-title">Checkpoints (Across Instances)</div>
            <div class="models-list" id="luna-checkpoints">
                ${panelState.checkpoints.length > 0
                    ? panelState.checkpoints.map(ckpt => `
                        <div class="model-item">
                            <div style="font-weight: 500;">${ckpt.name}</div>
                            <div style="font-size: 11px; opacity: 0.7; margin-top: 2px;">
                                ${ckpt.size_mb.toFixed(1)} MB • ${ckpt.device} • ${ckpt.dtype} • ${ckpt.instance_id}
                            </div>
                        </div>
                    `).join('')
                    : '<div style="opacity: 0.5">No checkpoints loaded</div>'
                }
            </div>
        </div>
        
        <div class="button-row">
            <button class="btn btn-refresh" id="luna-refresh">↻ Refresh</button>
            ${panelState.running 
                ? '<button class="btn btn-stop" id="luna-toggle">Stop</button>'
                : '<button class="btn btn-start" id="luna-toggle">Start Tray</button>'
            }
            ${panelState.error ? '<button class="btn btn-refresh" id="luna-reconnect" style="margin-left: auto;">⚡ Fix</button>' : ''}
        </div>
    `;
    
    return container;
}

function formatUptime(seconds) {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
}

function getVramPercent() {
    if (panelState.vramTotal === 0) return 0;
    return Math.min(100, (panelState.vramUsed / panelState.vramTotal) * 100);
}

async function fetchDaemonStatus() {
    try {
        const response = await api.fetchApi("/luna/daemon/status");
        if (response.ok) {
            const data = await response.json();
            
            // Debug: log data to see what we're getting
            if (data.gpus && data.gpus.length > 0) {
                console.log('[Luna Panel] GPU data:', data.gpus);
            }
            if (data.checkpoints && data.checkpoints.length > 0) {
                console.log('[Luna Panel] Checkpoints:', data.checkpoints);
            }
            
            panelState.running = data.running || false;
            panelState.error = data.error || null;
            panelState.device = data.device || "unknown";
            panelState.vramUsed = data.vram_used_gb || 0;
            panelState.vramAllocated = data.vram_allocated_gb || 0;
            panelState.vramTotal = data.vram_total_gb || 0;
            panelState.vramPercent = data.vram_percent || 0;
            panelState.requests = data.request_count || 0;
            panelState.clipRequests = data.clip_request_count || 0;
            panelState.vaeRequests = data.vae_request_count || 0;
            panelState.uptime = data.uptime_seconds || 0;
            panelState.modelsLoaded = data.models_loaded || [];
            panelState.checkpoints = data.checkpoints || [];
            panelState.gpus = data.gpus || [];
            panelState.vaeLoaded = data.vae_loaded || false;
            panelState.clipLoaded = data.clip_loaded || false;
            panelState.weightRegistryModels = data.weight_registry_models || [];
            panelState.comfyuiVram = data.comfyui_vram || null;
        } else {
            panelState.running = false;
            panelState.error = "API Error: " + response.status;
        }
    } catch (e) {
        panelState.running = false;
        panelState.error = "Connection Failed";
    }
}

async function reconnectDaemon() {
    try {
        panelState.error = "Attempting reconnect...";
        updatePanelUI();
        attachPanelEventListeners();  // Re-attach if error message changed
        
        const response = await api.fetchApi("/luna/daemon/reconnect", { method: "POST" });
        const data = await response.json();
        
        if (!response.ok || data.status !== "ok") {
            panelState.error = data.message || "Reconnect failed";
        } else {
            panelState.error = null;
            await fetchDaemonStatus();
        }
        
        updatePanelUI();
        attachPanelEventListeners();  // Re-attach after update
    } catch (e) {
        panelState.error = "Reconnect failed: " + e.message;
        updatePanelUI();
        attachPanelEventListeners();
        console.error("Reconnect error:", e);
    }
}

async function ensureDaemonRunning() {
    /**
     * Smart checklist workflow:
     * 1. Check if daemon is running
     * 2. If running + started: nothing to do
     * 3. If running + not started: start it
     * 4. If not running: run it first
     */
    
    console.log("[Luna] Starting daemon checklist...");
    
    // Step 1: Check current status
    await fetchDaemonStatus();
    console.log(`[Luna] Status check: running=${panelState.running}, error=${panelState.error}`);
    
    // Step 2: If already running and no error, we're done
    if (panelState.running && !panelState.error) {
        console.log("[Luna] ✓ Daemon is running");
        return true;
    }
    
    // Step 3: If running but showing error, try to reconnect
    if (panelState.running && panelState.error) {
        console.log("[Luna] Daemon running but error detected, reconnecting...");
        await reconnectDaemon();
        await fetchDaemonStatus();
        
        if (panelState.running && !panelState.error) {
            console.log("[Luna] ✓ Reconnection successful");
            return true;
        }
    }
    
    // Step 4: Not running, start it
    if (!panelState.running) {
        console.log("[Luna] Daemon not running, starting...");
        try {
            const response = await api.fetchApi("/luna/daemon/start", { method: "POST" });
            const data = await response.json();
            
            if (!response.ok) {
                panelState.error = data.message || `HTTP ${response.status}`;
                console.error(`[Luna] Start failed: ${panelState.error}`);
                updatePanelUI();
                attachPanelEventListeners();
                return false;
            }
            
            if (data.status === "error") {
                panelState.error = data.message || "Start failed";
                console.error(`[Luna] Start error: ${panelState.error}`);
                updatePanelUI();
                attachPanelEventListeners();
                return false;
            }
            
            console.log("[Luna] ✓ Daemon started successfully");
            panelState.error = null;
            await fetchDaemonStatus();
            updatePanelUI();
            attachPanelEventListeners();
            return true;
            
        } catch (e) {
            panelState.error = "Start failed: " + e.message;
            console.error(`[Luna] Exception during start: ${e.message}`);
            updatePanelUI();
            attachPanelEventListeners();
            return false;
        }
    }
    
    console.log("[Luna] Unexpected state, falling back to status update");
    updatePanelUI();
    attachPanelEventListeners();
    return panelState.running && !panelState.error;
}

async function toggleDaemon() {
    try {
        const endpoint = panelState.running ? "/luna/daemon/stop" : "/luna/daemon/start";
        const response = await api.fetchApi(endpoint, { method: "POST" });
        const data = await response.json();
        
        if (!response.ok) {
            panelState.error = data.message || `HTTP ${response.status}`;
            updatePanelUI();
            attachPanelEventListeners();  // Re-attach listeners in case error message changed
            return;
        }
        
        if (data.status === "error") {
            panelState.error = data.message || "Unknown error";
        } else {
            panelState.error = null;  // Clear error on success
        }
        
        await fetchDaemonStatus();
        updatePanelUI();
        attachPanelEventListeners();  // Re-attach after UI update
    } catch (e) {
        panelState.error = "Request failed: " + e.message;
        updatePanelUI();
        attachPanelEventListeners();
        console.error("Failed to toggle daemon:", e);
    }
}

async function unloadModels() {
    try {
        const response = await api.fetchApi("/luna/daemon/unload", { method: "POST" });
        
        if (!response.ok) {
            panelState.error = `Failed to unload: HTTP ${response.status}`;
            updatePanelUI();
            attachPanelEventListeners();
            return;
        }
        
        const data = await response.json();
        if (data.status === "ok") {
            panelState.error = null;  // Clear any previous errors
            await fetchDaemonStatus();
            updatePanelUI();
        } else {
            panelState.error = data.message || "Unload failed";
            updatePanelUI();
        }
        attachPanelEventListeners();  // Re-attach after any state change
    } catch (e) {
        panelState.error = "Failed to unload models: " + e.message;
        updatePanelUI();
        attachPanelEventListeners();
        console.error("Unload error:", e);
    }
}

async function loadDeviceConfig() {
    // Load available GPUs and current device configuration
    try {
        const response = await api.fetchApi("/luna/daemon/devices");
        if (response.ok) {
            const data = await response.json();
            return data;
        }
    } catch (e) {
        console.error("Failed to load device config:", e);
    }
    return null;
}

async function setClipDevice(device) {
    // Change CLIP device
    try {
        const response = await api.fetchApi("/luna/daemon/set-clip-device", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device })
        });
        
        if (!response.ok) {
            const data = await response.json();
            panelState.error = data.error || "Failed to set CLIP device";
            updatePanelUI();
            return false;
        }
        
        const data = await response.json();
        if (data.success) {
            panelState.error = null;
            await fetchDaemonStatus();
            await updateDeviceUI();
            return true;
        } else {
            panelState.error = data.error || "Failed to set CLIP device";
            updatePanelUI();
            return false;
        }
    } catch (e) {
        panelState.error = "Request failed: " + e.message;
        updatePanelUI();
        return false;
    }
}

async function setVaeDevice(device) {
    // Change VAE device
    try {
        const response = await api.fetchApi("/luna/daemon/set-vae-device", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device })
        });
        
        if (!response.ok) {
            const data = await response.json();
            panelState.error = data.error || "Failed to set VAE device";
            updatePanelUI();
            return false;
        }
        
        const data = await response.json();
        if (data.success) {
            panelState.error = null;
            await fetchDaemonStatus();
            await updateDeviceUI();
            return true;
        } else {
            panelState.error = data.error || "Failed to set VAE device";
            updatePanelUI();
            return false;
        }
    } catch (e) {
        panelState.error = "Request failed: " + e.message;
        updatePanelUI();
        return false;
    }
}

async function setLlmDevice(device) {
    // Change LLM device
    try {
        const response = await api.fetchApi("/luna/daemon/set-llm-device", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device })
        });
        
        if (!response.ok) {
            const data = await response.json();
            panelState.error = data.error || "Failed to set LLM device";
            updatePanelUI();
            return false;
        }
        
        const data = await response.json();
        if (data.success) {
            panelState.error = null;
            await fetchDaemonStatus();
            await updateDeviceUI();
            return true;
        } else {
            panelState.error = data.error || "Failed to set LLM device";
            updatePanelUI();
            return false;
        }
    } catch (e) {
        panelState.error = "Request failed: " + e.message;
        updatePanelUI();
        return false;
    }
}

async function updateDeviceUI() {
    // Update the device configuration UI with available GPUs
    const deviceConfig = document.getElementById("luna-device-config");
    if (!deviceConfig) return;
    
    try {
        const devices = await loadDeviceConfig();
        if (!devices || !devices.success) {
            deviceConfig.innerHTML = `<div style="opacity: 0.5">Unable to load device info</div>`;
            return;
        }
        
        const { devices: current, available_gpus } = devices;
        
        // Build HTML for device selectors
        let html = '<div style="display: flex; flex-direction: column; gap: 10px;">';
        
        // CLIP Device
        html += `
            <div>
                <div style="font-size: 11px; opacity: 0.7; margin-bottom: 4px;">CLIP Device</div>
                <select id="luna-clip-device-select" style="width: 100%; padding: 6px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--comfy-input-bg); color: var(--fg-color);">
                    ${available_gpus.map(gpu => `<option value="${gpu}" ${current.clip === gpu ? 'selected' : ''}>${gpu}</option>`).join('')}
                    <option value="cpu" ${current.clip === 'cpu' ? 'selected' : ''}>cpu</option>
                </select>
            </div>
        `;
        
        // VAE Device
        html += `
            <div>
                <div style="font-size: 11px; opacity: 0.7; margin-bottom: 4px;">VAE Device</div>
                <select id="luna-vae-device-select" style="width: 100%; padding: 6px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--comfy-input-bg); color: var(--fg-color);">
                    ${available_gpus.map(gpu => `<option value="${gpu}" ${current.vae === gpu ? 'selected' : ''}>${gpu}</option>`).join('')}
                    <option value="cpu" ${current.vae === 'cpu' ? 'selected' : ''}>cpu</option>
                </select>
            </div>
        `;
        
        // LLM Device
        html += `
            <div>
                <div style="font-size: 11px; opacity: 0.7; margin-bottom: 4px;">LLM Device</div>
                <select id="luna-llm-device-select" style="width: 100%; padding: 6px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--comfy-input-bg); color: var(--fg-color);">
                    ${available_gpus.map(gpu => `<option value="${gpu}" ${current.llm === gpu ? 'selected' : ''}>${gpu}</option>`).join('')}
                    <option value="cpu" ${current.llm === 'cpu' ? 'selected' : ''}>cpu</option>
                </select>
            </div>
        `;
        
        html += '</div>';
        
        deviceConfig.innerHTML = html;
        
        // Attach change listeners
        const clipSelect = deviceConfig.querySelector("#luna-clip-device-select");
        const vaeSelect = deviceConfig.querySelector("#luna-vae-device-select");
        const llmSelect = deviceConfig.querySelector("#luna-llm-device-select");
        
        if (clipSelect) {
            clipSelect.onchange = async (e) => {
                const success = await setClipDevice(e.target.value);
                if (success) {
                    console.log(`[Luna] CLIP device changed to ${e.target.value}`);
                }
            };
        }
        
        if (vaeSelect) {
            vaeSelect.onchange = async (e) => {
                const success = await setVaeDevice(e.target.value);
                if (success) {
                    console.log(`[Luna] VAE device changed to ${e.target.value}`);
                }
            };
        }
        
        if (llmSelect) {
            llmSelect.onchange = async (e) => {
                const success = await setLlmDevice(e.target.value);
                if (success) {
                    console.log(`[Luna] LLM device changed to ${e.target.value}`);
                }
            };
        }
    } catch (e) {
        deviceConfig.innerHTML = `<div style="opacity: 0.5; color: #f87171;">Error loading devices</div>`;
        console.error("Device UI error:", e);
    }
}

function attachPanelEventListeners() {
    const panel = document.querySelector(".luna-daemon-panel");
    if (!panel) return;
    
    // Refresh button
    const refreshBtn = panel.querySelector("#luna-refresh");
    if (refreshBtn) {
        refreshBtn.onclick = null;  // Clear any existing listener
        refreshBtn.onclick = async () => {
            refreshBtn.disabled = true;
            const originalText = refreshBtn.textContent;
            refreshBtn.textContent = "⟳ Refreshing...";
            try {
                await fetchDaemonStatus();
                updatePanelUI();
                attachPanelEventListeners();  // Re-attach after update
            } finally {
                refreshBtn.disabled = false;
                refreshBtn.textContent = originalText;
            }
        };
    }
    
    // Toggle (Start/Stop) button
    const toggleBtn = panel.querySelector("#luna-toggle");
    if (toggleBtn) {
        toggleBtn.onclick = null;  // Clear existing
        toggleBtn.onclick = async () => {
            toggleBtn.disabled = true;
            const wasRunning = panelState.running;
            const originalText = toggleBtn.textContent;
            
            try {
                if (wasRunning) {
                    // Stopping: use simple toggle
                    toggleBtn.textContent = "Stopping...";
                    await toggleDaemon();
                } else {
                    // Starting: use smart checklist
                    toggleBtn.textContent = "Ensuring daemon...";
                    const success = await ensureDaemonRunning();
                    if (success) {
                        console.log("[Luna] Daemon ready!");
                    }
                }
            } finally {
                toggleBtn.disabled = false;
                // Text will be updated by updatePanelUI
            }
        };
    }
    
    // Unload Models button
    const unloadBtn = panel.querySelector("#luna-unload");
    if (unloadBtn) {
        unloadBtn.onclick = null;  // Clear existing
        unloadBtn.onclick = async () => {
            unloadBtn.disabled = true;
            const originalText = unloadBtn.textContent;
            unloadBtn.textContent = "Unloading...";
            try {
                await unloadModels();
            } finally {
                unloadBtn.disabled = false;
                unloadBtn.textContent = originalText;
            }
        };
    }
    
    // Reconnect (Fix) button
    const reconnectBtn = panel.querySelector("#luna-reconnect");
    if (reconnectBtn) {
        reconnectBtn.onclick = null;  // Clear existing
        reconnectBtn.onclick = async () => {
            reconnectBtn.disabled = true;
            const originalText = reconnectBtn.textContent;
            reconnectBtn.textContent = "Connecting...";
            try {
                await reconnectDaemon();
            } finally {
                reconnectBtn.disabled = false;
                reconnectBtn.textContent = originalText;
            }
        };
    }
}

function updatePanelUI() {
    const panel = document.querySelector(".luna-daemon-panel");
    if (!panel) {
        console.warn("[Luna] Panel not found in DOM, skipping update");
        return;
    }
    
    // If GPU or checkpoint data exists, we need to recreate the entire panel
    // because those sections are created dynamically
    if (panelState.gpus.length > 0 || panelState.checkpoints.length > 0) {
        // Find the parent container and replace the panel content
        const parent = panel.parentElement;
        const newPanel = createPanelContent();
        parent.replaceChild(newPanel, panel);
        attachPanelEventListeners();
        return;
    }
    
    // Otherwise, do incremental updates for simple fields
    // Update status indicator
    const indicator = panel.querySelector(".status-indicator");
    if (indicator) {
        indicator.className = `status-indicator ${panelState.running ? 'running' : 'stopped'}`;
    }
    
    // Update text values
    const statusEl = document.getElementById("luna-status");
    if (statusEl) statusEl.textContent = panelState.running ? 'Running (System Tray)' : 'Stopped';
    
    const deviceEl = document.getElementById("luna-device");
    if (deviceEl) deviceEl.textContent = panelState.device;
    
    const requestsEl = document.getElementById("luna-requests");
    if (requestsEl) requestsEl.textContent = panelState.requests.toString();
    
    const clipRequestsEl = document.getElementById("luna-clip-requests");
    if (clipRequestsEl) clipRequestsEl.textContent = panelState.clipRequests.toString();
    
    const vaeRequestsEl = document.getElementById("luna-vae-requests");
    if (vaeRequestsEl) vaeRequestsEl.textContent = panelState.vaeRequests.toString();
    
    const uptimeEl = document.getElementById("luna-uptime");
    if (uptimeEl) uptimeEl.textContent = formatUptime(panelState.uptime);
    
    const vramEl = document.getElementById("luna-vram");
    if (vramEl) vramEl.textContent = `${panelState.vramUsed.toFixed(2)} / ${panelState.vramTotal.toFixed(1)} GB`;
    
    const vramAllocatedEl = document.getElementById("luna-vram-allocated");
    if (vramAllocatedEl) vramAllocatedEl.textContent = `${panelState.vramAllocated.toFixed(2)} GB`;
    
    const vramPercentEl = document.getElementById("luna-vram-percent");
    if (vramPercentEl) vramPercentEl.textContent = `${panelState.vramPercent.toFixed(1)}%`;
    
    // Update VRAM bar
    const vramBar = document.getElementById("luna-vram-bar");
    if (vramBar) vramBar.style.width = `${panelState.vramPercent}%`;
    
    // Update models list
    const modelsList = document.getElementById("luna-models");
    if (modelsList) {
        let modelsHtml;
        if (panelState.modelsLoaded && panelState.modelsLoaded.length > 0) {
            modelsHtml = panelState.modelsLoaded.map(m => `<div class="model-item">${m}</div>`).join('');
        } else {
            modelsHtml = '<div style="opacity: 0.5">Waiting for first workflow...</div>';
        }
        
        // Update models HTML
        modelsList.innerHTML = modelsHtml;
        
        // Handle unload button visibility
        const modelsSection = modelsList.parentElement;
        let unloadBtn = modelsSection.querySelector("#luna-unload");
        
        if (panelState.modelsLoaded.length > 0) {
            // Models are loaded - ensure unload button exists
            if (!unloadBtn) {
                unloadBtn = document.createElement("button");
                unloadBtn.id = "luna-unload";
                unloadBtn.className = "btn btn-unload";
                unloadBtn.style.marginTop = "8px";
                unloadBtn.style.width = "100%";
                unloadBtn.textContent = "Unload Models";
                modelsSection.appendChild(unloadBtn);
                unloadBtn.onclick = unloadModels;
            }
        } else {
            // No models - remove unload button if it exists
            if (unloadBtn) {
                unloadBtn.remove();
            }
        }
    }
    
    // Update device configuration UI
    if (panelState.running && !panelState.error) {
        updateDeviceUI();
    }
    
    // Update error message
    const errorMsg = panel.querySelector(".error-message");
    if (panelState.error) {
        if (!errorMsg) {
            const h3 = panel.querySelector("h3");
            if (h3) {
                const newError = document.createElement("div");
                newError.className = "error-message";
                newError.textContent = panelState.error;
                h3.parentNode.insertBefore(newError, h3.nextSibling);
            }
        } else {
            errorMsg.textContent = panelState.error;
            errorMsg.style.display = "block";
        }
    } else if (errorMsg) {
        errorMsg.style.display = "none";
    }
    
    // Update toggle button state and text
    const toggleBtn = document.getElementById("luna-toggle");
    if (toggleBtn) {
        if (panelState.running) {
            toggleBtn.className = "btn btn-stop";
            toggleBtn.textContent = "Stop";
        } else {
            toggleBtn.className = "btn btn-start";
            toggleBtn.textContent = "Start Tray";
        }
    }
    
    // Show/hide reconnect button
    const buttonRow = panel.querySelector(".button-row");
    if (buttonRow) {
        let reconnectBtn = buttonRow.querySelector("#luna-reconnect");
        if (panelState.error) {
            if (!reconnectBtn) {
                reconnectBtn = document.createElement("button");
                reconnectBtn.id = "luna-reconnect";
                reconnectBtn.className = "btn btn-refresh";
                reconnectBtn.style.marginLeft = "auto";
                reconnectBtn.textContent = "⚡ Fix";
                buttonRow.appendChild(reconnectBtn);
                reconnectBtn.onclick = reconnectDaemon;
            }
        } else if (reconnectBtn) {
            reconnectBtn.remove();
        }
    }
    
    // Re-attach all event listeners after DOM updates
    attachPanelEventListeners();
}

// Register the sidebar panel
app.registerExtension({
    name: "Luna.DaemonPanel",
    
    async setup() {
        // Fetch initial status
        await fetchDaemonStatus();
        
        // Auto-start daemon if not running using smart checklist
        if (!panelState.running) {
            console.log("[Luna] Auto-starting daemon...");
            try {
                const success = await ensureDaemonRunning();
                if (success) {
                    console.log("[Luna] Auto-start successful");
                } else {
                    console.warn("[Luna] Auto-start failed - daemon not responding");
                }
            } catch (e) {
                console.error("[Luna] Failed to auto-start daemon:", e);
            }
        }
        
        // Add sidebar tab
        app.ui.settings.addSetting({
            id: "Luna.DaemonPanel.enabled",
            name: "Enable Luna Daemon Panel",
            type: "boolean",
            defaultValue: true,
        });
        
        // Add to ComfyUI sidebar
        if (app.extensionManager?.registerSidebarTab) {
            app.extensionManager.registerSidebarTab({
                id: "luna-daemon",
                icon: "🌙",
                title: "Luna Daemon",
                tooltip: "Luna VAE/CLIP Daemon Control",
                type: "custom",
                render: (container) => {
                    container.innerHTML = "";  // Clear previous content
                    container.appendChild(createPanelContent());
                    attachPanelEventListeners();  // Attach all event listeners
                }
            });
        } else {
            // Fallback: Add to menu or as floating panel
            console.log("Luna Daemon Panel: Sidebar API not available, using menu fallback");
            
            // Add menu item
            const menu = document.querySelector(".comfy-menu");
            if (menu) {
                const menuItem = document.createElement("button");
                menuItem.textContent = "🌙 Daemon";
                menuItem.className = "comfy-menu-item";
                menuItem.onclick = () => {
                    showFloatingPanel();
                };
                menu.appendChild(menuItem);
            }
        }
        
        // Auto-refresh every 1 second for real-time VRAM monitoring
        setInterval(async () => {
            const panel = document.querySelector(".luna-daemon-panel");
            if (panel) {
                await fetchDaemonStatus();
                updatePanelUI();
            }
        }, 1000);
    }
});

// Floating panel fallback for older ComfyUI versions
function showFloatingPanel() {
    let panel = document.getElementById("luna-daemon-floating-panel");
    
    if (panel) {
        panel.style.display = panel.style.display === "none" ? "block" : "none";
        return;
    }
    
    panel = document.createElement("div");
    panel.id = "luna-daemon-floating-panel";
    panel.style.cssText = `
        position: fixed;
        top: 50px;
        right: 20px;
        width: 280px;
        background: var(--comfy-menu-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        z-index: 10000;
        max-height: 80vh;
        overflow-y: auto;
    `;
    
    const header = document.createElement("div");
    header.style.cssText = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        border-bottom: 1px solid var(--border-color);
        cursor: move;
    `;
    header.innerHTML = `
        <span style="font-weight: 600;">🌙 Luna Daemon</span>
        <button style="background: none; border: none; color: var(--fg-color); cursor: pointer; font-size: 18px;">&times;</button>
    `;
    header.querySelector("button").onclick = () => panel.style.display = "none";
    
    panel.appendChild(header);
    panel.appendChild(createPanelContent());
    document.body.appendChild(panel);
    
    // Attach event listeners
    attachPanelEventListeners();
    
    // Make draggable
    let isDragging = false;
    let dragOffset = { x: 0, y: 0 };
    
    header.addEventListener("mousedown", (e) => {
        if (e.target.tagName === "BUTTON") return;
        isDragging = true;
        dragOffset.x = e.clientX - panel.offsetLeft;
        dragOffset.y = e.clientY - panel.offsetTop;
    });
    
    document.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        panel.style.left = (e.clientX - dragOffset.x) + "px";
        panel.style.top = (e.clientY - dragOffset.y) + "px";
        panel.style.right = "auto";
    });
    
    document.addEventListener("mouseup", () => {
        isDragging = false;
    });
}
