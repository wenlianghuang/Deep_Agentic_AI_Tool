# Ollama 設置指南

本指南說明如何在 Deep Agentic AI Tool 中設置和使用 Ollama，特別是 Llama 3.2 3B 模型。

## 📋 前置需求

- macOS 或 Linux 系統
- 至少 16GB 記憶體（推薦）
- Python >= 3.13

## 🚀 安裝步驟

### 1. 安裝 Ollama

**macOS:**
```bash
brew install ollama
```

或從官網下載：https://ollama.com

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. 下載 Llama 3.2 模型

```bash
ollama pull llama3.2:3b
```

這會下載約 2GB 的模型文件。

### 3. 啟動 Ollama 服務

Ollama 通常會自動啟動，如果需要手動啟動：

```bash
ollama serve
```

服務預設運行在 `http://localhost:11434`

### 4. 驗證安裝

測試模型是否可用：

```bash
ollama run llama3.2:3b "Hello, how are you?"
```

## ⚙️ 配置專案

### 1. 更新環境變數

在專案根目錄的 `.env` 文件中添加：

```env
# 啟用 Ollama
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

### 2. 可選配置

如果需要使用其他 Ollama 模型，可以修改：

```env
OLLAMA_MODEL=qwen2.5:7b        # 使用 Qwen2.5
OLLAMA_MODEL=llama3.1:8b        # 使用 Llama 3.1
OLLAMA_MODEL=deepseek-r1:7b     # 使用 DeepSeek-R1
OLLAMA_MODEL=mistral:7b         # 使用 Mistral
```

## 🎯 使用方式

系統會按照以下優先順序自動選擇 LLM：

1. **Groq API**（如果配置了 `GROQ_API_KEY`）
2. **Ollama**（如果 `USE_OLLAMA=true` 且服務可用）
3. **MLX 模型**（備援選項）

當 Groq API 額度用完時，系統會自動切換到 Ollama（如果啟用），否則使用 MLX 模型。

## 🔍 檢查當前使用的模型

啟動應用後，查看控制台輸出：

- `✅ 使用 Groq API (優先)` - 使用 Groq API
- `✅ 使用 Ollama 模型 (llama3.2:3b)` - 使用 Ollama
- `ℹ️ 使用本地 MLX 模型` - 使用 MLX 模型

## 🐛 故障排除

### Ollama 服務無法連接

**問題：** `⚠️ Ollama 初始化失敗: Connection refused`

**解決方案：**
1. 確認 Ollama 服務正在運行：`ollama serve`
2. 檢查端口是否被占用：`lsof -i :11434`
3. 確認 `OLLAMA_BASE_URL` 配置正確

### 模型找不到

**問題：** `⚠️ Ollama 初始化失敗: model not found`

**解決方案：**
```bash
# 下載模型
ollama pull llama3.2:3b

# 列出已安裝的模型
ollama list
```

### 記憶體不足

**問題：** 系統運行緩慢或崩潰

**解決方案：**
- Llama 3.2:3B 需要約 2GB RAM
- 確保系統有足夠的可用記憶體（推薦至少 8GB）
- 這個模型已經很輕量，適合 16GB 記憶體的系統

## 📊 模型比較

| 模型 | 大小 | 記憶體需求 | 特點 |
|------|------|-----------|------|
| llama3.2:3b | ~2GB | ~4GB | 輕量高效，適合 16GB 記憶體系統，Meta 開源 |
| deepseek-r1:7b | ~4.7GB | ~8GB | 優秀的推理能力，適合數學、編程 |
| qwen2.5:7b | ~4.5GB | ~8GB | 通用能力強，中英文支援好 |
| llama3.1:8b | ~4.6GB | ~8GB | Meta 開源，性能穩定 |
| mistral:7b | ~4.1GB | ~7GB | 速度快，效率高 |

## 💡 性能優化建議

1. **優先使用 Groq API**：如果可用，Groq API 速度最快
2. **Ollama 作為備援**：當 Groq 不可用時，Ollama 提供良好的本地推理
3. **MLX 作為最後備援**：在 Apple Silicon 上，MLX 模型有硬體優化

## 📚 相關資源

- [Ollama 官方文檔](https://ollama.com/docs)
- [Llama 3.2 模型資訊](https://ollama.com/library/llama3.2)
- [LangChain Ollama 整合](https://python.langchain.com/docs/integrations/llms/ollama)

---

**注意**：首次使用時，Ollama 會下載模型文件，這可能需要一些時間，請耐心等待。

