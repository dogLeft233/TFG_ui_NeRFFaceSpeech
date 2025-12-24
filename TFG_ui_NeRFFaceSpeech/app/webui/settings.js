/**
 * 设置管理模块 - 用于 gradio_app 的 HTML 前端
 * 包含 API 地址管理和主题/字体设置功能
 */

// API 基础地址（优先级：URL参数 > localStorage > 默认值）
const API_BASE_URL = (() => {
    // 1. 优先从 URL 参数获取（支持端口转发场景）
    if (typeof window !== 'undefined') {
        const urlParams = new URLSearchParams(window.location.search);
        const apiUrlFromUrl = urlParams.get('api_url') || urlParams.get('apiUrl');
        if (apiUrlFromUrl) {
            // 保存到 localStorage 以便后续使用
            localStorage.setItem('API_BASE_URL', apiUrlFromUrl);
            return apiUrlFromUrl;
        }
    }

    // 2. 从 localStorage 获取
    if (typeof window !== 'undefined') {
        const saved = localStorage.getItem('API_BASE_URL');
        if (saved) return saved;
    }

    // 3. 默认值
    return 'http://localhost:8000';
})();

// 保存 API 地址到 localStorage
function setApiBaseUrl(url) {
    localStorage.setItem('API_BASE_URL', url);
    // 触发自定义事件，通知其他页面更新
    window.dispatchEvent(new CustomEvent('apiUrlChanged', { detail: url }));
}

// 获取 API 地址
function getApiBaseUrl() {
    if (typeof window !== 'undefined' && window.API_BASE_URL) {
        return window.API_BASE_URL;
    }
    const saved = localStorage.getItem('API_BASE_URL');
    return saved || 'http://localhost:8000';
}

// 获取设置 API 地址
function getSettingsApiBase() {
    const baseUrl = getApiBaseUrl();
    return `${baseUrl}/api/settings`;
}

// 默认设置（仅在数据库中没有设置时使用）
const DEFAULT_SETTINGS = {
    nerf_theme: "tech",
    nerf_font: "Inter",
    nerf_font_size: "medium",
    nerf_custom_font_size: "14"
};

/**
 * 从后端获取所有设置
 */
async function fetchSettings() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000); // 3秒超时

        const response = await fetch(getSettingsApiBase(), {
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        if (!result.success) {
            console.warn("⚠️ 获取设置失败，使用默认设置:", result.error || result.message);
            return DEFAULT_SETTINGS;
        }

        if (!result.data) {
            throw new Error("API返回数据格式错误：缺少data字段");
        }

        return result.data;
    } catch (error) {
        // 网络错误或其他错误，返回默认设置（允许页面加载）
        console.warn("⚠️ 获取设置失败（可能是网络问题），使用默认设置:", error);
        return DEFAULT_SETTINGS;
    }
}

/**
 * 获取单个设置值（从数据库读取）
 */
async function getSetting(key) {
    const settings = await fetchSettings();
    return settings[key] || DEFAULT_SETTINGS[key];
}

/**
 * 更新单个设置（直接写入数据库）
 */
async function updateSetting(key, value) {
    try {
        const response = await fetch(`${getSettingsApiBase()}/${key}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ "value": String(value) })
        });
        const result = await response.json();
        return result.success || false;
    } catch (error) {
        console.error("更新设置失败:", error);
        return false;
    }
}

/**
 * 批量更新设置（直接写入数据库）
 */
async function updateSettings(settings) {
    try {
        const response = await fetch(getSettingsApiBase(), {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(settings)
        });
        const result = await response.json();
        return result.success || false;
    } catch (error) {
        console.error("批量更新设置失败:", error);
        return false;
    }
}

/**
 * 应用主题到页面
 */
function applyThemeToPage(theme) {
    if (document.body) {
        document.body.className = 'theme-' + theme;
    } else {
        document.documentElement.setAttribute('data-pending-theme', theme);
        if (document.readyState === 'loading') {
            const applyThemeHandler = () => {
                if (document.body) {
                    document.body.className = 'theme-' + theme;
                    document.documentElement.removeAttribute('data-pending-theme');
                }
                document.removeEventListener('DOMContentLoaded', applyThemeHandler);
            };
            document.addEventListener('DOMContentLoaded', applyThemeHandler);
        }
    }
}

/**
 * 应用字体到页面
 */
function applyFontToPage(font) {
    if (document.body) {
        document.body.style.fontFamily = font;
    } else {
        document.documentElement.style.fontFamily = font;
    }
}

/**
 * 应用字体大小到页面
 */
function applyFontSizeToPage(size, customSize = null) {
    let fontSize;
    if (size === 'small') fontSize = '12px';
    else if (size === 'medium') fontSize = '14px';
    else if (size === 'large') fontSize = '16px';
    else if (size === 'custom') {
        fontSize = (customSize || DEFAULT_SETTINGS.nerf_custom_font_size || '14') + 'px';
    }
    if (fontSize) {
        document.documentElement.style.setProperty('--base-font-size', fontSize);
        document.documentElement.style.fontSize = fontSize;
    }
}

/**
 * 应用主题（会保存到数据库）
 */
async function applyTheme(theme) {
    applyThemeToPage(theme);
    await updateSetting('nerf_theme', theme);
    if (document.readyState === 'complete') {
        await updateThemeSelection();
    } else {
        document.addEventListener('DOMContentLoaded', async () => {
            await updateThemeSelection();
        }, { once: true });
    }
}

/**
 * 应用字体（会保存到数据库）
 */
async function applyFont(font) {
    applyFontToPage(font);
    await updateSetting('nerf_font', font);
    if (document.readyState === 'complete') {
        await updateFontSelection();
    } else {
        document.addEventListener('DOMContentLoaded', async () => {
            await updateFontSelection();
        }, { once: true });
    }
}

/**
 * 应用字体大小（会保存到数据库）
 */
async function applyFontSize(size) {
    let fontSize;
    let customValue = null;

    if (size === 'small') fontSize = '12px';
    else if (size === 'medium') fontSize = '14px';
    else if (size === 'large') fontSize = '16px';
    else if (size === 'custom') {
        const customInput = document.getElementById('customSizeValue');
        if (customInput && customInput.value) {
            customValue = customInput.value;
            fontSize = customValue + 'px';
            await updateSetting('nerf_custom_font_size', customValue);
        } else {
            return;
        }
    }

    if (fontSize) {
        applyFontSizeToPage(size, customValue);
        await updateSetting('nerf_font_size', size);
        if (document.readyState === 'complete') {
            await updateFontSizeSelection();
        } else {
            document.addEventListener('DOMContentLoaded', async () => {
                await updateFontSizeSelection();
            }, { once: true });
        }
    }
}

/**
 * 更新主题选择状态
 */
async function updateThemeSelection(settings = null) {
    if (!document.body) return;
    try {
        if (!settings) {
            settings = await fetchSettings();
        }
        const currentTheme = settings.nerf_theme || DEFAULT_SETTINGS.nerf_theme;
        const themeCards = document.querySelectorAll('[data-theme]');
        if (themeCards.length === 0) return;
        themeCards.forEach(card => {
            card.classList.toggle('selected', card.dataset.theme === currentTheme);
        });
    } catch (error) {
        console.error("更新主题选择状态失败:", error);
    }
}

/**
 * 更新字体选择状态
 */
async function updateFontSelection(settings = null) {
    if (!document.body) return;
    try {
        if (!settings) {
            settings = await fetchSettings();
        }
        const currentFont = settings.nerf_font || DEFAULT_SETTINGS.nerf_font;
        const fontCards = document.querySelectorAll('[data-font]');
        if (fontCards.length === 0) return;
        fontCards.forEach(card => {
            card.classList.toggle('selected', card.dataset.font === currentFont);
        });
    } catch (error) {
        console.error("更新字体选择状态失败:", error);
    }
}

/**
 * 更新字号选择状态
 */
async function updateFontSizeSelection(settings = null) {
    if (!document.body) return;
    try {
        if (!settings) {
            settings = await fetchSettings();
        }
        const currentSize = settings.nerf_font_size || DEFAULT_SETTINGS.nerf_font_size;
        const sizeButtons = document.querySelectorAll('[data-size]');
        if (sizeButtons.length > 0) {
            sizeButtons.forEach(btn => {
                btn.classList.toggle('selected', btn.dataset.size === currentSize);
            });
        }

        const customInput = document.getElementById('customSizeInput');
        if (customInput) {
            if (currentSize === 'custom') {
                customInput.style.display = 'block';
                const customValue = settings.nerf_custom_font_size || DEFAULT_SETTINGS.nerf_custom_font_size;
                const customValueInput = document.getElementById('customSizeValue');
                if (customValueInput) {
                    customValueInput.value = customValue;
                }
            } else {
                customInput.style.display = 'none';
            }
        }
    } catch (error) {
        console.error("更新字号选择状态失败:", error);
    }
}

/**
 * 从数据库读取设置并应用（用于除start.html外的所有页面）
 */
async function applySettingsOnly(settings = null) {
    try {
        if (!settings) {
            settings = await fetchSettings();
        }

        const theme = settings.nerf_theme || DEFAULT_SETTINGS.nerf_theme;
        applyThemeToPage(theme);

        const font = settings.nerf_font || DEFAULT_SETTINGS.nerf_font;
        if (document.body) {
            applyFontToPage(font);
        } else {
            document.documentElement.style.fontFamily = font;
        }

        const fontSize = settings.nerf_font_size || DEFAULT_SETTINGS.nerf_font_size;
        const customSize = settings.nerf_custom_font_size || DEFAULT_SETTINGS.nerf_custom_font_size;
        applyFontSizeToPage(fontSize, fontSize === 'custom' ? customSize : null);
    } catch (error) {
        console.error("从数据库读取设置失败:", error);
        const theme = DEFAULT_SETTINGS.nerf_theme;
        applyThemeToPage(theme);
        if (document.body) {
            applyFontToPage(DEFAULT_SETTINGS.nerf_font);
        } else {
            document.documentElement.style.fontFamily = DEFAULT_SETTINGS.nerf_font;
        }
        applyFontSizeToPage(DEFAULT_SETTINGS.nerf_font_size, DEFAULT_SETTINGS.nerf_font_size === 'custom' ? DEFAULT_SETTINGS.nerf_custom_font_size : null);
    }
}

// 导出到全局作用域
if (typeof window !== 'undefined') {
    window.API_BASE_URL = API_BASE_URL;
    window.setApiBaseUrl = setApiBaseUrl;
    window.getApiBaseUrl = getApiBaseUrl;
    window.fetchSettings = fetchSettings;
    window.getSetting = getSetting;
    window.updateSetting = updateSetting;
    window.updateSettings = updateSettings;
    window.applyTheme = applyTheme;
    window.applyFont = applyFont;
    window.applyFontSize = applyFontSize;
    window.applySettingsOnly = applySettingsOnly;
    window.updateThemeSelection = updateThemeSelection;
    window.updateFontSelection = updateFontSelection;
    window.updateFontSizeSelection = updateFontSizeSelection;
    window.applyThemeToPage = applyThemeToPage;
    window.applyFontToPage = applyFontToPage;
    window.applyFontSizeToPage = applyFontSizeToPage;

    // 在控制台打印后端地址信息（settings.js 加载时）
    console.log('[settings.js] 后端 API 地址:', API_BASE_URL);
    console.log('[settings.js] API 文档地址:', API_BASE_URL + '/docs');
}

