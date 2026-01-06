# Parlant SDK 安裝說明

## 問題

如果遇到 `ModuleNotFoundError: No module named 'parlant.sdk'` 錯誤，這是因為：

1. **PyPI 版本不完整**：從 PyPI 安裝的 `parlant` 包（版本 0.1.0）不包含 `sdk` 模組
2. **需要開發版本**：Parlant SDK 需要從 GitHub 安裝開發版本

## 解決方案

### 方法 1：使用 uv 安裝（推薦）

```bash
# 激活虛擬環境
source .venv/bin/activate

# 卸載舊版本（如果有的話）
uv pip uninstall parlant

# 安裝開發版本
uv pip install "git+https://github.com/emcie-co/parlant@develop"
```

### 方法 2：使用 pip 安裝

```bash
# 激活虛擬環境
source .venv/bin/activate

# 卸載舊版本（如果有的話）
pip uninstall parlant

# 安裝開發版本
pip install "git+https://github.com/emcie-co/parlant@develop"
```

### 方法 3：使用 uv sync（自動安裝）

如果 `pyproject.toml` 已更新為使用 GitHub 版本：

```bash
uv sync
```

## 驗證安裝

安裝完成後，驗證是否正確：

```bash
python3 -c "import parlant.sdk as p; print('✅ Parlant SDK 安裝成功')"
```

應該看到：
```
✅ Parlant SDK 安裝成功
```

## pyproject.toml 配置

項目已配置為從 GitHub 安裝開發版本：

```toml
dependencies = [
    # ... 其他依賴
    "parlant @ git+https://github.com/emcie-co/parlant@develop",
]
```

使用 `uv sync` 會自動安裝正確的版本。

## 常見問題

### Q: 為什麼不能從 PyPI 安裝？

A: PyPI 上的 `parlant` 包（0.1.0 版本）是舊版本，不包含 `sdk` 模組。需要使用 GitHub 上的開發版本。

### Q: 安裝後還是報錯怎麼辦？

A: 
1. 確認虛擬環境已激活
2. 確認安裝的是開發版本：`pip show parlant` 應該顯示版本為 `3.1.0a1` 或更高
3. 重新啟動 Python 解釋器或終端

### Q: 如何更新到最新版本？

A: 
```bash
uv pip install --upgrade "git+https://github.com/emcie-co/parlant@develop"
```

## 版本信息

- **PyPI 版本**：0.1.0（不包含 SDK）
- **GitHub 開發版本**：3.1.0a1+（包含完整 SDK）

---

**注意**：確保使用 GitHub 開發版本才能正常使用 Parlant SDK 的所有功能。


