/**
 * 设置管理模块 - 使用后端API与数据库交互
 * 所有页面共享此模块来管理设置
 */

// 自动检测服务器地址（从当前页面的URL获取）
const SETTINGS_API_BASE = (() => {
    const currentOrigin = window.location.origin;
    const currentHostname = window.location.hostname;
    const currentPort = window.location.port;
    const currentPath = window.location.pathname;
    const currentProtocol = window.location.protocol;

    // 如果当前端口是7860（前端服务器），则使用后端服务器（8000端口）
    if (currentPort === '7860') {
        const backendUrl = `${currentProtocol}//${currentHostname}:8000`;
        return `${backendUrl}/api/settings`;
    }

    // 如果路径包含 /webui/，说明是挂载在FastAPI下的
    if (currentPath.includes('/webui/')) {
        return `${currentOrigin}/api/settings`;
    }

    // 否则使用当前origin
    return `${currentOrigin}/api/settings`;
})();

// 默认设置（仅在数据库中没有设置时使用）
const DEFAULT_SETTINGS = {
    nerf_theme: "tech",
    nerf_font: "Inter",
    nerf_font_size: "medium",
    nerf_custom_font_size: "14"
};

/**
 * 从后端获取所有设置（数据库未初始化时会抛出错误，阻止网页打开）
 */
async function fetchSettings() {
    try {
        // 设置超时，避免前端页面一直加载
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000); // 3秒超时

        const response = await fetch(SETTINGS_API_BASE, {
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        // 如果数据库未初始化（success为false），抛出错误
        if (!result.success) {
            const errorMsg = result.message || result.error || "数据库未初始化";
            console.error("❌ 数据库错误:", errorMsg);
            // 显示错误提示并抛出异常，阻止页面加载
            alert(`❌ 数据库错误\n\n${errorMsg}\n\n请先运行 start.py 初始化数据库，然后再打开网页。`);
            throw new Error(errorMsg);
        }

        if (!result.data) {
            throw new Error("API返回数据格式错误：缺少data字段");
        }

        return result.data;
    } catch (error) {
        // 如果是数据库未初始化错误，阻止页面加载
        if (error.message && (error.message.includes("数据库未初始化") ||
            error.message.includes("数据库文件不存在") ||
            error.message.includes("数据库"))) {
            console.error("❌ 数据库未初始化，阻止页面加载:", error);
            // 错误已经在上面显示了，这里直接抛出
            throw error;
        }
        // 其他错误（如网络错误），记录日志但返回默认设置（允许页面加载，但功能可能受限）
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
 * 更新单个设置（直接写入数据库，不使用缓存）
 */
async function updateSetting(key, value) {
    try {
        // 后端API期望 embed=True，即 {"value": "xxx"} 格式
        const response = await fetch(`${SETTINGS_API_BASE}/${key}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ "value": String(value) })  // 发送 {"value": "xxx"} 格式
        });
        const result = await response.json();
        return result.success || false;
    } catch (error) {
        console.error("更新设置失败:", error);
        return false;
    }
}

/**
 * 批量更新设置（直接写入数据库，不使用缓存）
 */
async function updateSettings(settings) {
    try {
        const response = await fetch(SETTINGS_API_BASE, {
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
        // 如果body还没加载，先设置到documentElement，等body加载后再应用
        document.documentElement.setAttribute('data-pending-theme', theme);
        // 监听DOMContentLoaded来应用主题
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
    document.body.style.fontFamily = font;
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
    // 更新选择状态（延迟执行，确保DOM已加载）
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
    // 更新选择状态（延迟执行，确保DOM已加载）
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
            return; // 如果没有自定义值，不应用
        }
    }

    if (fontSize) {
        applyFontSizeToPage(size, customValue);
        await updateSetting('nerf_font_size', size);
        // 更新选择状态（延迟执行，确保DOM已加载）
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
 * 更新主题选择状态（从数据库读取当前设置）
 * @param {Object} settings - 可选的设置对象，如果提供则直接使用，否则从数据库读取
 */
async function updateThemeSelection(settings = null) {
    if (!document.body) return; // 如果body还没加载，直接返回
    try {
        if (!settings) {
            settings = await fetchSettings();
        }
        const currentTheme = settings.nerf_theme || DEFAULT_SETTINGS.nerf_theme;
        const themeCards = document.querySelectorAll('[data-theme]');
        if (themeCards.length === 0) return; // 如果没有找到元素，直接返回
        themeCards.forEach(card => {
            card.classList.toggle('selected', card.dataset.theme === currentTheme);
        });
    } catch (error) {
        console.error("更新主题选择状态失败:", error);
    }
}

/**
 * 更新字体选择状态（从数据库读取当前设置）
 * @param {Object} settings - 可选的设置对象，如果提供则直接使用，否则从数据库读取
 */
async function updateFontSelection(settings = null) {
    if (!document.body) return; // 如果body还没加载，直接返回
    try {
        if (!settings) {
            settings = await fetchSettings();
        }
        const currentFont = settings.nerf_font || DEFAULT_SETTINGS.nerf_font;
        const fontCards = document.querySelectorAll('[data-font]');
        if (fontCards.length === 0) return; // 如果没有找到元素，直接返回
        fontCards.forEach(card => {
            card.classList.toggle('selected', card.dataset.font === currentFont);
        });
    } catch (error) {
        console.error("更新字体选择状态失败:", error);
    }
}

/**
 * 更新字号选择状态（从数据库读取当前设置）
 * @param {Object} settings - 可选的设置对象，如果提供则直接使用，否则从数据库读取
 */
async function updateFontSizeSelection(settings = null) {
    if (!document.body) return; // 如果body还没加载，直接返回
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
 * 总是从数据库读取，不使用缓存
 * @param {Object} settings - 可选的设置对象，如果提供则直接使用，否则从数据库读取
 */
async function applySettingsOnly(settings = null) {
    try {
        // 如果没有提供设置，则从数据库读取
        if (!settings) {
            settings = await fetchSettings();
        }

        // 应用数据库中的设置
        const theme = settings.nerf_theme || DEFAULT_SETTINGS.nerf_theme;
        applyThemeToPage(theme);

        // 应用字体
        const font = settings.nerf_font || DEFAULT_SETTINGS.nerf_font;
        if (document.body) {
            applyFontToPage(font);
        } else {
            document.documentElement.style.fontFamily = font;
        }

        // 应用字体大小
        const fontSize = settings.nerf_font_size || DEFAULT_SETTINGS.nerf_font_size;
        const customSize = settings.nerf_custom_font_size || DEFAULT_SETTINGS.nerf_custom_font_size;
        applyFontSizeToPage(fontSize, fontSize === 'custom' ? customSize : null);
    } catch (error) {
        console.error("从数据库读取设置失败:", error);
        // 如果读取失败，使用默认设置
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

/**
 * 初始化设置（页面加载时调用）
 * 这个函数会从数据库获取设置并应用
 * 注意：只在开发者页面（start.html）的DOMContentLoaded中调用，其他页面使用 applySettingsOnly()
 */
async function initSettings() {
    try {
        // 获取所有设置（从数据库）
        const settings = await fetchSettings();

        // 使用数据库中的设置，如果数据库中没有则使用默认值（但不更新数据库）
        const theme = settings.nerf_theme || DEFAULT_SETTINGS.nerf_theme;
        applyThemeToPage(theme);

        // 应用字体
        const font = settings.nerf_font || DEFAULT_SETTINGS.nerf_font;
        if (document.body) {
            applyFontToPage(font);
        } else {
            document.documentElement.style.fontFamily = font;
        }

        // 应用字体大小
        const fontSize = settings.nerf_font_size || DEFAULT_SETTINGS.nerf_font_size;
        const customSize = settings.nerf_custom_font_size || DEFAULT_SETTINGS.nerf_custom_font_size;
        applyFontSizeToPage(fontSize, fontSize === 'custom' ? customSize : null);

        // 更新选择状态（此时DOM应该已经加载完成）- 使用已获取的设置，避免重复API调用
        await Promise.all([
            updateThemeSelection(settings),
            updateFontSelection(settings),
            updateFontSizeSelection(settings)
        ]);

        return settings;
    } catch (error) {
        console.error("初始化设置失败:", error);
        // 如果获取设置失败，使用默认设置（仅用于显示，不保存到数据库）
        const theme = DEFAULT_SETTINGS.nerf_theme;
        applyThemeToPage(theme);
        // 即使失败也尝试更新UI
        try {
            await updateThemeSelection();
            await updateFontSelection();
            await updateFontSizeSelection();
        } catch (e) {
            // 忽略UI更新错误
        }
        return DEFAULT_SETTINGS;
    }
}

// 注意：不在脚本加载时自动执行初始化
// start.html 应该在 DOMContentLoaded 时手动调用 initSettings()
// 其他页面应该调用 applySettingsOnly() 来应用缓存的设置
// 这样可以避免阻塞页面加载，即使 fetchSettings() 失败也不会影响页面渲染

