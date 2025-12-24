const API_BASE = "http://localhost:8000";

const tabs = {
  infer: document.getElementById("tab-infer"),
  chat: document.getElementById("tab-chat"),
  train: document.getElementById("tab-train"),
};

const sections = {
  infer: document.getElementById("section-infer"),
  chat: document.getElementById("section-chat"),
  train: document.getElementById("section-train"),
};

const statusBox = document.getElementById("status");

function setStatus(msg) {
  statusBox.textContent = msg;
}

function switchTab(name) {
  Object.keys(sections).forEach((k) => {
    sections[k].classList.toggle("hidden", k !== name);
    tabs[k].classList.toggle("active", k === name);
  });
}

tabs.infer.onclick = () => switchTab("infer");
tabs.chat.onclick = () => switchTab("chat");
tabs.train.onclick = () => switchTab("train");

// ---------------- Models ----------------
const modelSelect = document.getElementById("model");
const refreshBtn = document.getElementById("refresh");

async function loadModels() {
  setStatus("正在加载模型列表...");
  try {
    const res = await fetch(`${API_BASE}/models`);
    const data = await res.json();
    
    // 检查是否是错误响应
    if (data.success === false || (data.error && !Array.isArray(data))) {
      const errorMsg = data.error || "未知错误";
      setStatus(`❌ 模型列表获取失败：${errorMsg}`);
      modelSelect.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "模型加载失败";
      opt.disabled = true;
      modelSelect.appendChild(opt);
      return;
    }
    
    // 正常情况：data 应该是数组
    if (!Array.isArray(data) || data.length === 0) {
      setStatus("❌ 模型列表为空，请检查模型目录");
      modelSelect.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "模型列表为空";
      opt.disabled = true;
      modelSelect.appendChild(opt);
      return;
    }
    
    modelSelect.innerHTML = "";
    data.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      modelSelect.appendChild(opt);
    });
    setStatus("模型列表已更新");
  } catch (e) {
    setStatus(`❌ 加载模型列表失败：${e.message || e}`);
    modelSelect.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "加载失败";
    opt.disabled = true;
    modelSelect.appendChild(opt);
  }
}
refreshBtn.onclick = loadModels;

// ---------------- Inference ----------------
const textInput = document.getElementById("text");
const characterSelect = document.getElementById("character");
const runBtn = document.getElementById("run");
const videoEl = document.getElementById("video");

function toHttpVideoPath(localPath) {
  // localPath like /root/autodl-tmp/NeRFFaceSpeech_Code/outputs/video/<id>/output_NeRFFaceSpeech.mp4
  const marker = "/outputs/video/";
  const idx = localPath.indexOf(marker);
  if (idx === -1) return localPath;
  const sub = localPath.slice(idx + marker.length); // <id>/output...
  return `${API_BASE}/videos/${sub}`;
}

async function runInference() {
  const text = textInput.value.trim();
  const character = characterSelect.value;
  const model = modelSelect.value;
  if (!text) {
    setStatus("请输入文本");
    return;
  }
  runBtn.disabled = true;
  runBtn.textContent = "生成中...";
  setStatus("请求后端生成中...");
  try {
    const res = await fetch(`${API_BASE}/generate_video`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, character, model_name: model }),
    });
    const data = await res.json();
    if (!data.success) {
      setStatus("生成失败：" + JSON.stringify(data));
      videoEl.src = "";
    } else {
      const url = toHttpVideoPath(data.video_url);
      videoEl.src = url;
      videoEl.load();
      setStatus("✅ 生成成功");
    }
  } catch (e) {
    setStatus("请求异常：" + e);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "开始生成视频";
  }
}
runBtn.onclick = runInference;

// ---------------- Chat (placeholder) ----------------
const chatBtn = document.getElementById("chat-btn");
const chatText = document.getElementById("chat-text");
const chatAudio = document.getElementById("chat-audio");
chatBtn.onclick = () => {
  setStatus("实时对话接口未接入，请后端开放后绑定。");
  chatText.value = "前端占位：待接入实时对话 API";
  chatAudio.src = "";
};

// ---------------- Train (placeholder) ----------------
const trainBtn = document.getElementById("train-btn");
const trainLog = document.getElementById("train-log");
trainBtn.onclick = () => {
  setStatus("训练接口未接入，请后端开放后绑定。");
  trainLog.value = "前端占位：待接入训练 API";
};

// ---------------- Help toggle ----------------
const helpBtn = document.getElementById("help-btn");
const accordion = document.getElementById("help-accordion");
helpBtn.onclick = () => {
  accordion.classList.toggle("show");
};

// init
switchTab("infer");
loadModels();

