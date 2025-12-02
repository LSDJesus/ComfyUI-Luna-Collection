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
    device: "unknown",
    vramUsed: 0,
    vramTotal: 0,
    requests: 0,
    uptime: 0,
    modelsLoaded: [],
    vaeLoaded: false,
    clipLoaded: false,
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
        
        <div class="section">
            <div class="section-title">Status</div>
            <div class="stat-row">
                <span class="stat-label">State</span>
                <span class="stat-value" id="luna-status">${panelState.running ? 'Running' : 'Stopped'}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Device</span>
                <span class="stat-value" id="luna-device">${panelState.device}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Requests</span>
                <span class="stat-value" id="luna-requests">${panelState.requests}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Uptime</span>
                <span class="stat-value" id="luna-uptime">${formatUptime(panelState.uptime)}</span>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">VRAM</div>
            <div class="stat-row">
                <span class="stat-label">Usage</span>
                <span class="stat-value" id="luna-vram">${panelState.vramUsed.toFixed(1)} / ${panelState.vramTotal.toFixed(1)} GB</span>
            </div>
            <div class="vram-bar">
                <div class="vram-fill" id="luna-vram-bar" style="width: ${getVramPercent()}%"></div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Loaded Models</div>
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
        
        <div class="button-row">
            <button class="btn btn-refresh" id="luna-refresh">↻ Refresh</button>
            ${panelState.running 
                ? '<button class="btn btn-stop" id="luna-toggle">Stop</button>'
                : '<button class="btn btn-start" id="luna-toggle">Start</button>'
            }
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
            panelState.running = data.running || false;
            panelState.device = data.device || "unknown";
            panelState.vramUsed = data.vram_used_gb || 0;
            panelState.vramTotal = data.vram_total_gb || 0;
            panelState.requests = data.request_count || 0;
            panelState.uptime = data.uptime_seconds || 0;
            panelState.modelsLoaded = data.models_loaded || [];
            panelState.vaeLoaded = data.vae_loaded || false;
            panelState.clipLoaded = data.clip_loaded || false;
        } else {
            panelState.running = false;
        }
    } catch (e) {
        panelState.running = false;
    }
}

async function toggleDaemon() {
    try {
        const endpoint = panelState.running ? "/luna/daemon/stop" : "/luna/daemon/start";
        await api.fetchApi(endpoint, { method: "POST" });
        await fetchDaemonStatus();
        updatePanelUI();
    } catch (e) {
        console.error("Failed to toggle daemon:", e);
    }
}

async function unloadModels() {
    try {
        const response = await api.fetchApi("/luna/daemon/unload", { method: "POST" });
        const data = await response.json();
        if (data.status === "ok") {
            await fetchDaemonStatus();
            updatePanelUI();
        } else {
            console.error("Failed to unload models:", data.message);
        }
    } catch (e) {
        console.error("Failed to unload models:", e);
    }
}

function updatePanelUI() {
    const panel = document.querySelector(".luna-daemon-panel");
    if (!panel) return;
    
    // Update status indicator
    const indicator = panel.querySelector(".status-indicator");
    indicator.className = `status-indicator ${panelState.running ? 'running' : 'stopped'}`;
    
    // Update text values
    document.getElementById("luna-status").textContent = panelState.running ? 'Running' : 'Stopped';
    document.getElementById("luna-device").textContent = panelState.device;
    document.getElementById("luna-requests").textContent = panelState.requests;
    document.getElementById("luna-uptime").textContent = formatUptime(panelState.uptime);
    document.getElementById("luna-vram").textContent = 
        `${panelState.vramUsed.toFixed(1)} / ${panelState.vramTotal.toFixed(1)} GB`;
    
    // Update VRAM bar
    document.getElementById("luna-vram-bar").style.width = `${getVramPercent()}%`;
    
    // Update models list
    const modelsList = document.getElementById("luna-models");
    if (modelsList) {
        let modelsHtml = panelState.modelsLoaded.length > 0 
            ? panelState.modelsLoaded.map(m => `<div class="model-item">${m}</div>`).join('')
            : '<div style="opacity: 0.5">Waiting for first workflow...</div>';
        
        // Add unload button if models are loaded
        if (panelState.modelsLoaded.length > 0) {
            // Check if unload button exists, if not we need to rebuild section
            if (!document.getElementById("luna-unload")) {
                const section = modelsList.parentElement;
                section.innerHTML = `
                    <div class="section-title">Loaded Models</div>
                    <div class="models-list" id="luna-models">${modelsHtml}</div>
                    <button class="btn btn-unload" id="luna-unload" style="margin-top: 8px; width: 100%;">Unload Models</button>
                `;
                section.querySelector("#luna-unload")?.addEventListener("click", unloadModels);
            } else {
                modelsList.innerHTML = modelsHtml;
            }
        } else {
            modelsList.innerHTML = modelsHtml;
            // Remove unload button if exists
            document.getElementById("luna-unload")?.remove();
        }
    }
    
    // Update toggle button
    const toggleBtn = document.getElementById("luna-toggle");
    if (panelState.running) {
        toggleBtn.className = "btn btn-stop";
        toggleBtn.textContent = "Stop";
    } else {
        toggleBtn.className = "btn btn-start";
        toggleBtn.textContent = "Start";
    }
}

// Register the sidebar panel
app.registerExtension({
    name: "Luna.DaemonPanel",
    
    async setup() {
        // Fetch initial status
        await fetchDaemonStatus();
        
        // Add sidebar tab
        app.ui.settings.addSetting({
            id: "Luna.DaemonPanel.enabled",
            name: "Enable Luna Daemon Panel",
            type: "boolean",
            defaultValue: true,
        });
        
        // Create sidebar panel using ComfyUI's sidebar API
        const sidebarContent = createPanelContent();
        
        // Add to ComfyUI sidebar
        if (app.extensionManager?.registerSidebarTab) {
            app.extensionManager.registerSidebarTab({
                id: "luna-daemon",
                icon: "🌙",
                title: "Luna Daemon",
                tooltip: "Luna VAE/CLIP Daemon Control",
                type: "custom",
                render: (container) => {
                    container.appendChild(createPanelContent());
                    
                    // Add event listeners
                    container.querySelector("#luna-refresh")?.addEventListener("click", async () => {
                        await fetchDaemonStatus();
                        updatePanelUI();
                    });
                    
                    container.querySelector("#luna-toggle")?.addEventListener("click", toggleDaemon);
                    container.querySelector("#luna-unload")?.addEventListener("click", unloadModels);
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
        
        // Auto-refresh every 5 seconds
        setInterval(async () => {
            await fetchDaemonStatus();
            updatePanelUI();
        }, 5000);
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
    
    // Add event listeners
    panel.querySelector("#luna-refresh")?.addEventListener("click", async () => {
        await fetchDaemonStatus();
        updatePanelUI();
    });
    
    panel.querySelector("#luna-toggle")?.addEventListener("click", toggleDaemon);
    panel.querySelector("#luna-unload")?.addEventListener("click", unloadModels);
    
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
