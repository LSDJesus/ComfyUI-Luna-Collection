import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

/**
 * Luna Connections Manager - Sidebar widget for managing LoRA/Embedding connections
 * Provides a visual interface to link LoRAs and embeddings to wildcard categories/tags
 */

const LUNA_CONNECTIONS_STYLES = `
.luna-connections-panel {
    position: fixed;
    right: 0;
    top: 50px;
    width: 380px;
    height: calc(100vh - 60px);
    background: #1a1a1a;
    border-left: 2px solid #4a9eff;
    z-index: 9999;
    display: none;
    flex-direction: column;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    box-shadow: -4px 0 20px rgba(0,0,0,0.5);
}

.luna-connections-panel.visible {
    display: flex;
}

.luna-panel-header {
    background: linear-gradient(135deg, #2d5aa8 0%, #1a3d7c 100%);
    padding: 12px 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #4a9eff;
}

.luna-panel-title {
    color: #fff;
    font-weight: 600;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.luna-panel-title::before {
    content: "🔗";
}

.luna-close-btn {
    background: rgba(255,255,255,0.1);
    border: none;
    color: #fff;
    width: 28px;
    height: 28px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    transition: background 0.2s;
}

.luna-close-btn:hover {
    background: rgba(255,255,255,0.2);
}

.luna-panel-tabs {
    display: flex;
    background: #222;
    border-bottom: 1px solid #333;
}

.luna-tab {
    flex: 1;
    padding: 10px;
    background: transparent;
    border: none;
    color: #888;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.2s;
    border-bottom: 2px solid transparent;
}

.luna-tab:hover {
    color: #ccc;
    background: rgba(255,255,255,0.05);
}

.luna-tab.active {
    color: #4a9eff;
    border-bottom-color: #4a9eff;
    background: rgba(74, 158, 255, 0.1);
}

.luna-panel-content {
    flex: 1;
    overflow-y: auto;
    padding: 15px;
}

.luna-section {
    margin-bottom: 20px;
}

.luna-section-title {
    color: #4a9eff;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #333;
}

.luna-search-box {
    width: 100%;
    padding: 10px 12px;
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 6px;
    color: #fff;
    font-size: 13px;
    margin-bottom: 12px;
    box-sizing: border-box;
}

.luna-search-box:focus {
    outline: none;
    border-color: #4a9eff;
    box-shadow: 0 0 0 2px rgba(74, 158, 255, 0.2);
}

.luna-item-list {
    max-height: 200px;
    overflow-y: auto;
    background: #222;
    border-radius: 6px;
    border: 1px solid #333;
}

.luna-item {
    padding: 10px 12px;
    border-bottom: 1px solid #333;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: background 0.2s;
}

.luna-item:last-child {
    border-bottom: none;
}

.luna-item:hover {
    background: rgba(74, 158, 255, 0.1);
}

.luna-item.selected {
    background: rgba(74, 158, 255, 0.2);
    border-left: 3px solid #4a9eff;
}

.luna-item-name {
    color: #ddd;
    font-size: 12px;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.luna-item-badge {
    background: #4a9eff;
    color: #fff;
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 10px;
    margin-left: 8px;
}

.luna-item-badge.unlinked {
    background: #666;
}

.luna-form-group {
    margin-bottom: 15px;
}

.luna-label {
    display: block;
    color: #aaa;
    font-size: 11px;
    font-weight: 500;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.luna-input, .luna-select, .luna-textarea {
    width: 100%;
    padding: 10px 12px;
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 6px;
    color: #fff;
    font-size: 13px;
    box-sizing: border-box;
}

.luna-textarea {
    min-height: 80px;
    resize: vertical;
    font-family: monospace;
}

.luna-input:focus, .luna-select:focus, .luna-textarea:focus {
    outline: none;
    border-color: #4a9eff;
}

.luna-tag-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}

.luna-tag {
    background: #333;
    color: #ccc;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid #444;
}

.luna-tag:hover {
    background: #444;
}

.luna-tag.selected {
    background: #4a9eff;
    color: #fff;
    border-color: #4a9eff;
}

.luna-tag .remove {
    margin-left: 6px;
    opacity: 0.6;
}

.luna-tag .remove:hover {
    opacity: 1;
}

.luna-btn {
    padding: 10px 16px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.2s;
}

.luna-btn-primary {
    background: #4a9eff;
    color: #fff;
}

.luna-btn-primary:hover {
    background: #3a8eef;
}

.luna-btn-secondary {
    background: #444;
    color: #ccc;
}

.luna-btn-secondary:hover {
    background: #555;
}

.luna-btn-danger {
    background: #dc3545;
    color: #fff;
}

.luna-btn-danger:hover {
    background: #c82333;
}

.luna-btn-group {
    display: flex;
    gap: 10px;
    margin-top: 15px;
}

.luna-stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
}

.luna-stat-card {
    background: #2a2a2a;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    border: 1px solid #333;
}

.luna-stat-value {
    font-size: 24px;
    font-weight: 600;
    color: #4a9eff;
}

.luna-stat-label {
    font-size: 10px;
    color: #888;
    text-transform: uppercase;
    margin-top: 4px;
}

.luna-model-filter {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}

.luna-model-chip {
    padding: 5px 10px;
    background: #333;
    border: 1px solid #444;
    border-radius: 15px;
    color: #aaa;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
}

.luna-model-chip:hover {
    background: #444;
}

.luna-model-chip.active {
    background: #4a9eff;
    border-color: #4a9eff;
    color: #fff;
}

.luna-toggle-btn {
    /* Toolbar-docked button style */
    background: linear-gradient(135deg, #4a9eff 0%, #2d5aa8 100%);
    border: none;
    border-radius: 4px;
    color: #fff;
    font-size: 16px;
    cursor: pointer;
    padding: 6px 10px;
    transition: transform 0.2s, box-shadow 0.2s;
}

.luna-toggle-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 2px 10px rgba(74, 158, 255, 0.4);
}

.luna-toast {
    position: fixed;
    bottom: 20px;
    right: 20px;
    padding: 12px 20px;
    background: #333;
    color: #fff;
    border-radius: 8px;
    font-size: 13px;
    z-index: 10000;
    animation: slideIn 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.luna-toast.success {
    background: #28a745;
}

.luna-toast.error {
    background: #dc3545;
}

@keyframes slideIn {
    from { transform: translateX(100px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.luna-weight-slider {
    display: flex;
    align-items: center;
    gap: 10px;
}

.luna-weight-slider input[type="range"] {
    flex: 1;
    height: 6px;
    -webkit-appearance: none;
    background: #333;
    border-radius: 3px;
}

.luna-weight-slider input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: #4a9eff;
    border-radius: 50%;
    cursor: pointer;
}

.luna-weight-value {
    min-width: 40px;
    text-align: center;
    color: #4a9eff;
    font-weight: 500;
}

.luna-empty-state {
    text-align: center;
    padding: 30px;
    color: #666;
}

.luna-empty-state-icon {
    font-size: 48px;
    margin-bottom: 15px;
}
`;

class LunaConnectionsManager {
    constructor() {
        this.panel = null;
        this.toggleBtn = null;
        this.currentTab = 'loras';
        this.selectedItem = null;
        this.connections = { loras: {}, embeddings: {} };
        this.allLoras = [];
        this.allEmbeddings = [];
        this.allTags = [];
        this.allCategories = [];
        this.modelFilter = 'all';
        this.usingSidebar = false;
        
        this.init();
    }
    
    async init() {
        // Inject styles
        const styleEl = document.createElement('style');
        styleEl.textContent = LUNA_CONNECTIONS_STYLES;
        document.head.appendChild(styleEl);
        
        // Try to register with ComfyUI sidebar first
        if (this.registerSidebarTab()) {
            this.usingSidebar = true;
        } else {
            // Fallback: Add to ComfyUI menu/toolbar
            this.createToolbarButton();
            this.createPanel();
        }
        
        // Load initial data
        await this.loadData();
    }
    
    registerSidebarTab() {
        // Try ComfyUI's sidebar API
        if (app.extensionManager?.registerSidebarTab) {
            try {
                app.extensionManager.registerSidebarTab({
                    id: "luna-connections",
                    icon: "🔗",
                    title: "Luna Connections",
                    tooltip: "Manage LoRA & Embedding Connections",
                    type: "custom",
                    render: (container) => {
                        this.renderSidebarContent(container);
                    }
                });
                console.log("[Luna Connections] Registered with sidebar API");
                return true;
            } catch (e) {
                console.log("[Luna Connections] Sidebar registration failed:", e);
                return false;
            }
        }
        return false;
    }
    
    // Get the active container (sidebar body or panel content)
    getActiveContainer() {
        if (this.usingSidebar) {
            return document.querySelector('#luna-sidebar-body');
        }
        return this.panel?.querySelector('#luna-panel-content') || this.panel;
    }
    
    renderSidebarContent(container) {
        // Create sidebar-friendly content
        container.innerHTML = "";
        container.style.cssText = "padding: 0; height: 100%; display: flex; flex-direction: column;";
        
        const content = document.createElement("div");
        content.className = "luna-sidebar-content";
        content.innerHTML = `
            <style>
                .luna-sidebar-content {
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    font-size: 13px;
                    color: var(--fg-color, #ddd);
                }
                .luna-sidebar-content .luna-panel-tabs {
                    display: flex;
                    border-bottom: 1px solid var(--border-color, #333);
                    flex-shrink: 0;
                }
                .luna-sidebar-content .luna-tab {
                    flex: 1;
                    padding: 10px;
                    background: transparent;
                    border: none;
                    color: var(--fg-color, #888);
                    cursor: pointer;
                    font-size: 12px;
                    transition: all 0.2s;
                }
                .luna-sidebar-content .luna-tab:hover {
                    background: var(--comfy-input-bg, #333);
                }
                .luna-sidebar-content .luna-tab.active {
                    color: #4a9eff;
                    border-bottom: 2px solid #4a9eff;
                    background: var(--comfy-input-bg, #252525);
                }
                .luna-sidebar-content .luna-panel-body {
                    flex: 1;
                    overflow-y: auto;
                    padding: 10px;
                }
            </style>
            <div class="luna-panel-tabs">
                <button class="luna-tab active" data-tab="loras">LoRAs</button>
                <button class="luna-tab" data-tab="embeddings">Embeds</button>
                <button class="luna-tab" data-tab="stats">Stats</button>
            </div>
            <div class="luna-panel-body" id="luna-sidebar-body">
                <!-- Content injected dynamically -->
            </div>
        `;
        
        container.appendChild(content);
        
        // Tab click handlers
        content.querySelectorAll('.luna-tab').forEach(tab => {
            tab.onclick = () => {
                content.querySelectorAll('.luna-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this.currentTab = tab.dataset.tab;
                this.selectedItem = null;
                this.renderSidebarTabContent(content.querySelector('#luna-sidebar-body'));
            };
        });
        
        // Initial render
        this.renderSidebarTabContent(content.querySelector('#luna-sidebar-body'));
    }
    
    renderSidebarTabContent(container) {
        switch (this.currentTab) {
            case 'loras':
                container.innerHTML = this.renderLorasTab();
                break;
            case 'embeddings':
                container.innerHTML = this.renderEmbeddingsTab();
                break;
            case 'stats':
                container.innerHTML = this.renderStatsTab();
                break;
        }
        this.attachEventListeners(container);
    }
    
    createToolbarButton() {
        // Try to find ComfyUI's menu/toolbar
        const menu = document.querySelector(".comfy-menu");
        const queue = document.querySelector("#queue-button");
        
        this.toggleBtn = document.createElement('button');
        this.toggleBtn.className = 'luna-toggle-btn';
        this.toggleBtn.innerHTML = '🔗';
        this.toggleBtn.title = 'Luna Connections Manager';
        this.toggleBtn.onclick = () => this.togglePanel();
        
        if (queue && queue.parentElement) {
            // Insert near queue button in toolbar
            queue.parentElement.insertBefore(this.toggleBtn, queue.nextSibling);
            console.log("[Luna Connections] Added to toolbar");
        } else if (menu) {
            // Fallback to menu
            menu.appendChild(this.toggleBtn);
            console.log("[Luna Connections] Added to menu");
        } else {
            // Last resort: fixed position button
            this.toggleBtn.style.cssText = `
                position: fixed;
                right: 10px;
                top: 55px;
                z-index: 9998;
                width: 40px;
                height: 40px;
                border-radius: 50%;
            `;
            document.body.appendChild(this.toggleBtn);
            console.log("[Luna Connections] Using floating button fallback");
        }
    }
    
    createToggleButton() {
        // Deprecated - now using createToolbarButton or sidebar
        this.createToolbarButton();
    }
    
    createPanel() {
        this.panel = document.createElement('div');
        this.panel.className = 'luna-connections-panel';
        this.panel.innerHTML = `
            <div class="luna-panel-header">
                <div class="luna-panel-title">Luna Connections</div>
                <button class="luna-close-btn" onclick="lunaConnections.togglePanel()">×</button>
            </div>
            
            <div class="luna-panel-tabs">
                <button class="luna-tab active" data-tab="loras">LoRAs</button>
                <button class="luna-tab" data-tab="embeddings">Embeddings</button>
                <button class="luna-tab" data-tab="stats">Stats</button>
            </div>
            
            <div class="luna-panel-content" id="luna-panel-content">
                <!-- Content injected dynamically -->
            </div>
        `;
        
        document.body.appendChild(this.panel);
        
        // Tab click handlers
        this.panel.querySelectorAll('.luna-tab').forEach(tab => {
            tab.onclick = () => this.switchTab(tab.dataset.tab);
        });
    }
    
    togglePanel() {
        this.panel.classList.toggle('visible');
        if (this.panel.classList.contains('visible')) {
            this.loadData();
        }
    }
    
    async loadData() {
        try {
            // Load connections
            const connResp = await api.fetchApi('/luna/connections/list');
            if (connResp.ok) {
                const data = await connResp.json();
                this.allTags = data.tags || [];
                this.allCategories = data.categories || [];
                
                // Load full connection configs
                const fullResp = await api.fetchApi('/luna/connections/full');
                if (fullResp.ok) {
                    const fullData = await fullResp.json();
                    this.connections = fullData;
                }
            }
            
            // Load available LoRAs
            const loraResp = await api.fetchApi('/object_info/LunaLoRAStacker');
            if (loraResp.ok) {
                const data = await loraResp.json();
                this.allLoras = data?.LunaLoRAStacker?.input?.optional?.lora_1?.[0] || [];
            }
            
            // Load available embeddings
            const embResp = await api.fetchApi('/embeddings');
            if (embResp.ok) {
                this.allEmbeddings = await embResp.json();
            }
            
        } catch (e) {
            console.error('[Luna Connections] Error loading data:', e);
        }
        
        this.renderCurrentTab();
    }
    
    switchTab(tabName) {
        this.currentTab = tabName;
        this.selectedItem = null;
        
        this.panel.querySelectorAll('.luna-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });
        
        this.renderCurrentTab();
    }
    
    renderCurrentTab() {
        const content = this.panel.querySelector('#luna-panel-content');
        
        switch (this.currentTab) {
            case 'loras':
                content.innerHTML = this.renderLorasTab();
                break;
            case 'embeddings':
                content.innerHTML = this.renderEmbeddingsTab();
                break;
            case 'stats':
                content.innerHTML = this.renderStatsTab();
                break;
        }
        
        this.attachEventListeners();
    }
    
    renderLorasTab() {
        const linkedLoras = Object.keys(this.connections.loras || {});
        
        return `
            <div class="luna-section">
                <div class="luna-section-title">Model Type Filter</div>
                <div class="luna-model-filter">
                    ${['all', 'sdxl', 'pony', 'illustrious', 'sd15', 'flux'].map(type => `
                        <span class="luna-model-chip ${this.modelFilter === type ? 'active' : ''}" 
                              data-filter="${type}">${type.toUpperCase()}</span>
                    `).join('')}
                </div>
            </div>
            
            <div class="luna-section">
                <div class="luna-section-title">Select LoRA</div>
                <input type="text" class="luna-search-box" placeholder="Search LoRAs..." id="lora-search">
                <div class="luna-item-list" id="lora-list">
                    ${this.allLoras.slice(0, 50).map(lora => `
                        <div class="luna-item ${this.selectedItem === lora ? 'selected' : ''}" 
                             data-name="${lora}">
                            <span class="luna-item-name">${this.formatLoraName(lora)}</span>
                            <span class="luna-item-badge ${linkedLoras.includes(lora) ? '' : 'unlinked'}">
                                ${linkedLoras.includes(lora) ? 'Linked' : 'Unlinked'}
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            ${this.selectedItem ? this.renderConnectionEditor('lora') : `
                <div class="luna-empty-state">
                    <div class="luna-empty-state-icon">👆</div>
                    <p>Select a LoRA to edit its connections</p>
                </div>
            `}
        `;
    }
    
    renderEmbeddingsTab() {
        const linkedEmbs = Object.keys(this.connections.embeddings || {});
        
        return `
            <div class="luna-section">
                <div class="luna-section-title">Model Type Filter</div>
                <div class="luna-model-filter">
                    ${['all', 'sdxl', 'sd15'].map(type => `
                        <span class="luna-model-chip ${this.modelFilter === type ? 'active' : ''}" 
                              data-filter="${type}">${type.toUpperCase()}</span>
                    `).join('')}
                </div>
            </div>
            
            <div class="luna-section">
                <div class="luna-section-title">Select Embedding</div>
                <input type="text" class="luna-search-box" placeholder="Search embeddings..." id="emb-search">
                <div class="luna-item-list" id="emb-list">
                    ${this.allEmbeddings.slice(0, 50).map(emb => `
                        <div class="luna-item ${this.selectedItem === emb ? 'selected' : ''}" 
                             data-name="${emb}">
                            <span class="luna-item-name">${emb}</span>
                            <span class="luna-item-badge ${linkedEmbs.includes(emb) ? '' : 'unlinked'}">
                                ${linkedEmbs.includes(emb) ? 'Linked' : 'Unlinked'}
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            ${this.selectedItem ? this.renderConnectionEditor('embedding') : `
                <div class="luna-empty-state">
                    <div class="luna-empty-state-icon">👆</div>
                    <p>Select an embedding to edit its connections</p>
                </div>
            `}
        `;
    }
    
    renderConnectionEditor(type) {
        const collection = type === 'lora' ? 'loras' : 'embeddings';
        const config = this.connections[collection]?.[this.selectedItem] || {};
        
        return `
            <div class="luna-section">
                <div class="luna-section-title">Edit: ${this.formatLoraName(this.selectedItem)}</div>
                
                <div class="luna-form-group">
                    <label class="luna-label">Trigger Words</label>
                    <input type="text" class="luna-input" id="conn-triggers" 
                           value="${(config.triggers || []).join(', ')}"
                           placeholder="comma, separated, triggers">
                </div>
                
                <div class="luna-form-group">
                    <label class="luna-label">Categories (YAML paths)</label>
                    <textarea class="luna-textarea" id="conn-categories" 
                              placeholder="clothing:lingerie.types&#10;pose:seductive">${(config.categories || []).join('\n')}</textarea>
                </div>
                
                <div class="luna-form-group">
                    <label class="luna-label">Tags</label>
                    <div class="luna-tag-container" id="tag-container">
                        ${(config.tags || []).map(tag => `
                            <span class="luna-tag selected" data-tag="${tag}">
                                ${tag} <span class="remove">×</span>
                            </span>
                        `).join('')}
                    </div>
                    <input type="text" class="luna-input" id="new-tag" 
                           placeholder="Add tag and press Enter" style="margin-top: 8px;">
                </div>
                
                <div class="luna-form-group">
                    <label class="luna-label">Model Type</label>
                    <select class="luna-select" id="conn-model-type">
                        ${['any', 'sdxl', 'pony', 'illustrious', 'sd15', 'flux'].map(mt => `
                            <option value="${mt}" ${config.model_type === mt ? 'selected' : ''}>
                                ${mt.toUpperCase()}
                            </option>
                        `).join('')}
                    </select>
                </div>
                
                <div class="luna-form-group">
                    <label class="luna-label">Default Weight: <span id="weight-display">${config.weight_default || 1.0}</span></label>
                    <div class="luna-weight-slider">
                        <input type="range" min="0" max="2" step="0.05" 
                               value="${config.weight_default || 1.0}" id="conn-weight">
                    </div>
                </div>
                
                <div class="luna-form-group">
                    <label class="luna-label">Notes</label>
                    <textarea class="luna-textarea" id="conn-notes" 
                              placeholder="Optional notes...">${config.notes || ''}</textarea>
                </div>
                
                <div class="luna-btn-group">
                    <button class="luna-btn luna-btn-primary" id="save-connection">💾 Save</button>
                    <button class="luna-btn luna-btn-danger" id="delete-connection">🗑️ Remove</button>
                </div>
            </div>
        `;
    }
    
    renderStatsTab() {
        const loraCount = Object.keys(this.connections.loras || {}).length;
        const embCount = Object.keys(this.connections.embeddings || {}).length;
        
        // Count by model type
        const byType = {};
        Object.values(this.connections.loras || {}).forEach(c => {
            const t = c.model_type || 'unknown';
            byType[t] = (byType[t] || 0) + 1;
        });
        
        return `
            <div class="luna-section">
                <div class="luna-section-title">Connection Statistics</div>
                <div class="luna-stats-grid">
                    <div class="luna-stat-card">
                        <div class="luna-stat-value">${loraCount}</div>
                        <div class="luna-stat-label">Linked LoRAs</div>
                    </div>
                    <div class="luna-stat-card">
                        <div class="luna-stat-value">${embCount}</div>
                        <div class="luna-stat-label">Linked Embeddings</div>
                    </div>
                    <div class="luna-stat-card">
                        <div class="luna-stat-value">${this.allTags.length}</div>
                        <div class="luna-stat-label">Unique Tags</div>
                    </div>
                    <div class="luna-stat-card">
                        <div class="luna-stat-value">${this.allCategories.length}</div>
                        <div class="luna-stat-label">Categories</div>
                    </div>
                </div>
            </div>
            
            <div class="luna-section">
                <div class="luna-section-title">LoRAs by Model Type</div>
                ${Object.entries(byType).map(([type, count]) => `
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333;">
                        <span style="color: #aaa; text-transform: uppercase; font-size: 11px;">${type}</span>
                        <span style="color: #4a9eff; font-weight: 600;">${count}</span>
                    </div>
                `).join('')}
            </div>
            
            <div class="luna-section">
                <div class="luna-section-title">Popular Tags</div>
                <div class="luna-tag-container">
                    ${this.allTags.slice(0, 20).map(tag => `
                        <span class="luna-tag">${tag}</span>
                    `).join('')}
                </div>
            </div>
            
            <div class="luna-btn-group">
                <button class="luna-btn luna-btn-secondary" id="refresh-data">🔄 Refresh</button>
                <button class="luna-btn luna-btn-secondary" id="export-json">📥 Export JSON</button>
            </div>
        `;
    }
    
    attachEventListeners(container = null) {
        const root = container || this.panel;
        if (!root) return;
        
        // Model filter chips
        root.querySelectorAll('.luna-model-chip').forEach(chip => {
            chip.onclick = () => {
                this.modelFilter = chip.dataset.filter;
                if (this.usingSidebar && container) {
                    this.renderSidebarTabContent(container);
                } else {
                    this.renderCurrentTab();
                }
            };
        });
        
        // Item selection
        root.querySelectorAll('.luna-item').forEach(item => {
            item.onclick = () => {
                this.selectedItem = item.dataset.name;
                if (this.usingSidebar && container) {
                    this.renderSidebarTabContent(container);
                } else {
                    this.renderCurrentTab();
                }
            };
        });
        
        // Search
        const loraSearch = root.querySelector('#lora-search');
        if (loraSearch) {
            loraSearch.oninput = (e) => this.filterList('lora-list', e.target.value, this.allLoras, root);
        }
        
        const embSearch = root.querySelector('#emb-search');
        if (embSearch) {
            embSearch.oninput = (e) => this.filterList('emb-list', e.target.value, this.allEmbeddings, root);
        }
        
        // Weight slider
        const weightSlider = root.querySelector('#conn-weight');
        if (weightSlider) {
            weightSlider.oninput = () => {
                root.querySelector('#weight-display').textContent = weightSlider.value;
            };
        }
        
        // Tag input
        const tagInput = root.querySelector('#new-tag');
        if (tagInput) {
            tagInput.onkeypress = (e) => {
                if (e.key === 'Enter' && tagInput.value.trim()) {
                    this.addTag(tagInput.value.trim());
                    tagInput.value = '';
                }
            };
        }
        
        // Tag removal
        root.querySelectorAll('.luna-tag .remove').forEach(btn => {
            btn.onclick = (e) => {
                e.stopPropagation();
                const tag = btn.parentElement.dataset.tag;
                btn.parentElement.remove();
            };
        });
        
        // Save button
        const saveBtn = root.querySelector('#save-connection');
        if (saveBtn) {
            saveBtn.onclick = () => this.saveConnection();
        }
        
        // Delete button
        const deleteBtn = root.querySelector('#delete-connection');
        if (deleteBtn) {
            deleteBtn.onclick = () => this.deleteConnection();
        }
        
        // Refresh button
        const refreshBtn = root.querySelector('#refresh-data');
        if (refreshBtn) {
            refreshBtn.onclick = () => this.loadData();
        }
        
        // Export button
        const exportBtn = root.querySelector('#export-json');
        if (exportBtn) {
            exportBtn.onclick = () => this.exportConnections();
        }
    }
    
    filterList(listId, query, items, container = null) {
        const root = container || this.panel;
        const list = root.querySelector(`#${listId}`);
        if (!list) return;
        
        const filtered = items.filter(item => 
            item.toLowerCase().includes(query.toLowerCase())
        ).slice(0, 50);
        
        const linkedItems = this.currentTab === 'loras' 
            ? Object.keys(this.connections.loras || {})
            : Object.keys(this.connections.embeddings || {});
        
        list.innerHTML = filtered.map(item => `
            <div class="luna-item ${this.selectedItem === item ? 'selected' : ''}" 
                 data-name="${item}">
                <span class="luna-item-name">${this.formatLoraName(item)}</span>
                <span class="luna-item-badge ${linkedItems.includes(item) ? '' : 'unlinked'}">
                    ${linkedItems.includes(item) ? 'Linked' : 'Unlinked'}
                </span>
            </div>
        `).join('');
        
        // Reattach click handlers
        list.querySelectorAll('.luna-item').forEach(item => {
            item.onclick = () => {
                this.selectedItem = item.dataset.name;
                this.renderCurrentTab();
            };
        });
    }
    
    addTag(tag) {
        const container = this.getActiveContainer()?.querySelector('#tag-container');
        if (!container) return;
        
        const tagEl = document.createElement('span');
        tagEl.className = 'luna-tag selected';
        tagEl.dataset.tag = tag;
        tagEl.innerHTML = `${tag} <span class="remove">×</span>`;
        tagEl.querySelector('.remove').onclick = () => tagEl.remove();
        container.appendChild(tagEl);
    }
    
    async saveConnection() {
        const type = this.currentTab === 'loras' ? 'lora' : 'embedding';
        const root = this.getActiveContainer();
        if (!root) return;
        
        // Gather form data
        const triggers = root.querySelector('#conn-triggers')?.value
            .split(',').map(t => t.trim()).filter(t => t) || [];
        
        const categories = root.querySelector('#conn-categories')?.value
            .split('\n').map(c => c.trim()).filter(c => c) || [];
        
        const tags = Array.from(root.querySelectorAll('#tag-container .luna-tag'))
            .map(el => el.dataset.tag);
        
        const config = {
            triggers,
            categories,
            tags,
            model_type: root.querySelector('#conn-model-type')?.value || 'any',
            weight_default: parseFloat(root.querySelector('#conn-weight')?.value || '1.0'),
            notes: root.querySelector('#conn-notes')?.value || ''
        };
        
        try {
            const resp = await api.fetchApi('/luna/connections/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type,
                    name: this.selectedItem,
                    config
                })
            });
            
            if (resp.ok) {
                this.showToast('Connection saved successfully!', 'success');
                await this.loadData();
            } else {
                throw new Error(await resp.text());
            }
        } catch (e) {
            this.showToast('Error saving: ' + e.message, 'error');
        }
    }
    
    async deleteConnection() {
        if (!confirm(`Remove connection for "${this.selectedItem}"?`)) return;
        
        const type = this.currentTab === 'loras' ? 'lora' : 'embedding';
        
        try {
            const resp = await api.fetchApi('/luna/connections/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type, name: this.selectedItem })
            });
            
            if (resp.ok) {
                this.showToast('Connection removed', 'success');
                this.selectedItem = null;
                await this.loadData();
            }
        } catch (e) {
            this.showToast('Error: ' + e.message, 'error');
        }
    }
    
    exportConnections() {
        const dataStr = JSON.stringify(this.connections, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'luna_connections_export.json';
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('Connections exported!', 'success');
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `luna-toast ${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(() => toast.remove(), 3000);
    }
    
    formatLoraName(name) {
        return name
            .replace('.safetensors', '')
            .replace('.pt', '')
            .replace(/_/g, ' ')
            .replace(/\//g, ' / ');
    }
}

// Initialize when app is ready
app.registerExtension({
    name: "luna.connectionsManager",
    async setup() {
        // Wait for app to be ready
        setTimeout(() => {
            window.lunaConnections = new LunaConnectionsManager();
        }, 1000);
    }
});
