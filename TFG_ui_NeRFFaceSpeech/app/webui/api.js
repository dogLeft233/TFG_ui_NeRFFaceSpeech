/**
 * API 客户端工具
 * 统一管理前端与后端的 HTTP 请求
 * 
 * 注意：API_BASE_URL 应该在 settings.js 中定义
 * 如果 settings.js 未加载，则使用默认值
 */

// 不在这里声明 API_BASE_URL，使用 settings.js 中的定义
// 如果 settings.js 未加载，则从 localStorage 或默认值获取

/**
 * 获取当前的 API 基础地址（动态获取，支持 settings.js 更新）
 * @returns {string} API 基础地址
 */
function getApiBaseUrl() {
    if (typeof window !== 'undefined' && window.API_BASE_URL) {
        return window.API_BASE_URL;
    }
    const saved = localStorage.getItem('API_BASE_URL');
    return saved || 'http://localhost:8000';
}

/**
 * 通用 HTTP 请求函数
 * @param {string} endpoint - API 端点（例如：'/models'）
 * @param {object} options - fetch 选项
 * @returns {Promise<Response>}
 */
async function apiRequest(endpoint, options = {}) {
    const baseUrl = getApiBaseUrl();
    const url = `${baseUrl}${endpoint}`;

    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
        // 默认超时时间：5秒
        signal: AbortSignal.timeout(5000),
    };

    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...(options.headers || {}),
        },
        // 如果 options 中有 signal，使用它；否则使用默认的超时 signal
        signal: options.signal || defaultOptions.signal,
    };

    try {
        const response = await fetch(url, mergedOptions);
        return response;
    } catch (error) {
        console.error(`API 请求失败 [${endpoint}]:`, error);
        throw error;
    }
}

/**
 * GET 请求
 * @param {string} endpoint - API 端点
 * @param {object} params - URL 查询参数（可选）
 * @returns {Promise<any>} JSON 响应数据
 */
async function apiGet(endpoint, params = {}) {
    let url = endpoint;

    // 添加查询参数
    if (Object.keys(params).length > 0) {
        const queryString = new URLSearchParams(params).toString();
        url += `?${queryString}`;
    }

    const response = await apiRequest(url, {
        method: 'GET',
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
}

/**
 * POST 请求
 * @param {string} endpoint - API 端点
 * @param {object} data - 请求体数据
 * @param {object} options - 额外的请求选项（可选）
 * @returns {Promise<any>} JSON 响应数据
 */
async function apiPost(endpoint, data = {}, options = {}) {
    const response = await apiRequest(endpoint, {
        method: 'POST',
        body: JSON.stringify(data),
        ...options, // 允许覆盖默认选项，如超时时间
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
}

/**
 * PUT 请求
 * @param {string} endpoint - API 端点
 * @param {object} data - 请求体数据
 * @returns {Promise<any>} JSON 响应数据
 */
async function apiPut(endpoint, data = {}) {
    const response = await apiRequest(endpoint, {
        method: 'PUT',
        body: JSON.stringify(data),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
}

/**
 * DELETE 请求
 * @param {string} endpoint - API 端点
 * @returns {Promise<any>} JSON 响应数据
 */
async function apiDelete(endpoint) {
    const response = await apiRequest(endpoint, {
        method: 'DELETE',
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
}

/**
 * 检查后端连接状态
 * @returns {Promise<boolean>} 是否连接成功
 */
async function checkBackendConnection() {
    try {
        const baseUrl = getApiBaseUrl();
        const response = await fetch(`${baseUrl}/docs`, {
            method: 'HEAD',
            mode: 'cors',
            cache: 'no-cache',
            signal: AbortSignal.timeout(3000), // 3秒超时
        });
        return response.ok;
    } catch (error) {
        // 如果 HEAD 请求失败，尝试 GET 请求
        try {
            const response = await fetch(`${API_BASE_URL}/docs`, {
                method: 'GET',
                mode: 'cors',
                cache: 'no-cache',
                signal: AbortSignal.timeout(3000),
            });
            return response.ok;
        } catch (e) {
            return false;
        }
    }
}

// ==================== 具体的 API 方法 ====================

/**
 * 获取模型列表
 * @returns {Promise<string[]>} 模型名称列表
 */
async function getModels() {
    const data = await apiGet('/models');
    // 后端可能返回 { success: true, data: [...] } 或直接返回数组
    if (data.success !== undefined) {
        return data.data || [];
    }
    return Array.isArray(data) ? data : [];
}

/**
 * 生成视频
 * @param {object} params - 生成参数
 * @param {string} params.text - 输入文本
 * @param {string} params.character - 角色名称
 * @param {string} params.model_name - 模型名称
 * @returns {Promise<object>} 生成结果
 */
async function generateVideo(params) {
    return await apiPost('/generate_video', params);
}

/**
 * 查询视频生成状态
 * @param {string} taskId - 任务ID
 * @returns {Promise<object>} 任务状态
 */
async function getVideoGenerationStatus(taskId) {
    return await apiGet(`/generate_video/status/${taskId}`);
}

/**
 * 更新聊天消息的视频路径
 * @param {string} messageId - 消息ID
 * @param {string} videoPath - 视频路径
 * @returns {Promise<object>} 更新结果
 */
async function updateMessageVideoPath(messageId, videoPath) {
    return await apiPut(`/chat/messages/${messageId}/video_path`, { video_path: videoPath });
}

/**
 * 对话接口
 * @param {object} params - 对话参数
 * @param {string} params.text - 用户输入的文本（可选）
 * @param {string} params.audio_base64 - Base64编码的音频数据（可选）
 * @param {string} params.character - 角色名称
 * @param {boolean} params.enable_audio - 是否生成音频回复
 * @param {string} params.session_id - 会话ID（可选）
 * @returns {Promise<object>} 对话结果
 */
async function chat(params) {
    return await apiPost('/chat', params);
}

/**
 * ASR语音识别接口
 * @param {object} params - ASR参数
 * @param {string} params.audio_base64 - Base64编码的音频数据（可选）
 * @param {string} params.audio_path - 音频文件路径（可选）
 * @param {string} params.model_name - Whisper模型名称（默认: base）
 * @param {string} params.language - 语言代码（可选，None表示自动检测）
 * @param {string} params.task - 任务类型（transcribe或translate，默认: transcribe）
 * @returns {Promise<object>} 识别结果
 */
async function transcribeAudio(params) {
    // 语音识别可能需要较长时间，特别是首次加载模型时，设置5分钟超时
    return await apiPost('/asr/transcribe', params, {
        signal: AbortSignal.timeout(300000) // 5分钟超时（300秒）
    });
}

/**
 * 检查ASR服务健康状态
 * @returns {Promise<object>} ASR服务状态
 */
async function checkASRHealth() {
    return await apiGet('/asr/health');
}

/**
 * 获取训练数据集列表
 * @returns {Promise<string[]>} 数据集路径列表
 */
async function getTrainingDatasets() {
    const data = await apiGet('/train/datasets');
    if (data.success !== undefined) {
        return data.data || [];
    }
    return Array.isArray(data) ? data : [];
}

/**
 * 启动训练任务
 * @param {object} params - 训练参数
 * @returns {Promise<object>} 训练任务信息
 */
async function startTraining(params) {
    return await apiPost('/train/start', params);
}

/**
 * 查询训练任务状态
 * @param {string} taskId - 任务ID
 * @returns {Promise<object>} 任务状态
 */
async function getTrainingStatus(taskId) {
    return await apiGet(`/train/status/${taskId}`);
}

/**
 * 获取训练任务列表
 * @returns {Promise<object[]>} 任务列表
 */
async function getTrainingTasks() {
    const data = await apiGet('/train/tasks');
    if (data.success !== undefined) {
        return data.data || [];
    }
    return Array.isArray(data) ? data : [];
}

/**
 * 停止训练任务
 * @param {string} taskId - 任务ID
 * @returns {Promise<object>} 操作结果
 */
async function stopTraining(taskId) {
    return await apiPost(`/train/stop/${taskId}`);
}

/**
 * 提交角色训练任务（文件上传）
 * @param {FormData} formData - 包含视频文件和训练参数的 FormData
 * @returns {Promise<object>} 训练任务信息
 */
async function trainCharacter(formData) {
    const baseUrl = getApiBaseUrl();
    const url = `${baseUrl}/character/train`;

    const response = await fetch(url, {
        method: 'POST',
        body: formData,
        // 不要设置 Content-Type，让浏览器自动设置（包含 boundary）
        signal: AbortSignal.timeout(60000), // 文件上传可能需要更长时间，60秒超时
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
}

/**
 * 查询角色训练任务状态
 * @param {string} taskId - 任务ID
 * @returns {Promise<object>} 任务状态
 */
async function getCharacterTrainingStatus(taskId) {
    return await apiGet(`/character/train/status/${taskId}`);
}

/**
 * 获取所有已训练的角色列表
 * @returns {Promise<string[]>} 角色名称列表
 */
async function getCharacters() {
    const data = await apiGet('/character/list');
    // 后端返回格式: { success: true, characters: [...] }
    if (data.success !== undefined) {
        // 只返回已训练完成的角色（有图像和音频的）
        const characters = (data.characters || []).filter(char =>
            char.exists && char.num_images > 0 && char.audio_exists
        );
        return characters.map(char => char.character_name);
    }
    return [];
}

// ==================== 导出 ====================

// 如果是在浏览器环境中，将函数挂载到 window 对象
// 注意：不覆盖 window.API_BASE_URL（如果 settings.js 已定义）
if (typeof window !== 'undefined') {
    // 只有在 window.API_BASE_URL 不存在时才设置（使用默认值）
    if (!window.API_BASE_URL) {
        window.API_BASE_URL = 'http://localhost:8000';
    }

    window.apiRequest = apiRequest;
    window.apiGet = apiGet;
    window.apiPost = apiPost;
    window.apiPut = apiPut;
    window.apiDelete = apiDelete;
    window.checkBackendConnection = checkBackendConnection;

    // 具体的 API 方法
    window.getModels = getModels;
    window.generateVideo = generateVideo;
    window.getVideoGenerationStatus = getVideoGenerationStatus;
    window.updateMessageVideoPath = updateMessageVideoPath;
    window.chat = chat;
    window.transcribeAudio = transcribeAudio;
    window.checkASRHealth = checkASRHealth;
    window.getTrainingDatasets = getTrainingDatasets;
    window.startTraining = startTraining;
    window.getTrainingStatus = getTrainingStatus;
    window.getTrainingTasks = getTrainingTasks;
    window.stopTraining = stopTraining;
    window.trainCharacter = trainCharacter;
    window.getCharacterTrainingStatus = getCharacterTrainingStatus;
    window.getCharacters = getCharacters;

    // 创建一个函数来获取当前的 API_BASE_URL（动态获取，支持 settings.js 更新）
    window.getApiBaseUrl = function () {
        if (window.API_BASE_URL) {
            return window.API_BASE_URL;
        }
        const saved = localStorage.getItem('API_BASE_URL');
        return saved || 'http://localhost:8000';
    };

    // 在控制台打印后端地址信息（api.js 加载时）
    const currentApiUrl = window.getApiBaseUrl();
    console.log('[api.js] 后端 API 地址:', currentApiUrl);
    console.log('[api.js] API 文档地址:', currentApiUrl + '/docs');
}

// 如果是在 Node.js 环境中（用于测试），使用 module.exports
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        API_BASE_URL,
        apiRequest,
        apiGet,
        apiPost,
        apiPut,
        apiDelete,
        checkBackendConnection,
        getModels,
        generateVideo,
        getVideoGenerationStatus,
        chat,
        getTrainingDatasets,
        startTraining,
        getTrainingStatus,
        getTrainingTasks,
        stopTraining,
        trainCharacter,
        getCharacterTrainingStatus,
        getCharacters,
    };
}

