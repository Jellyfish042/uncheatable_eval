# BioRxiv爬虫使用说明

## 概述

已成功创建BioRxiv爬虫，支持爬取BioRxiv生物学预印本论文。

## 功能特性

- **统一subtitle**: 所有BioRxiv论文使用统一的category `biorxiv_biorxiv`
- **分类记录**: 在metadata的`classification`字段中记录具体学科分类（如neuroscience、cell_biology等）
- **全类别支持**: 使用`classification="all"`参数爬取所有学科分类
- **日期范围**: 支持通过日期范围筛选论文
- **PDF处理**: 
  - 自动大小检查（默认50MB上限）
  - 使用MinerU进行OCR
  - 文本清理（移除参考文献、致谢等）

## 文件说明

### 新增文件

1. **`crawlers/crawler/spiders/biorxiv.py`** - BioRxiv爬虫主文件
2. **`crawlers/crawler/items.py`** - 添加了`BiorxivPaperItem`类
3. **`crawlers/crawler/policy.py`** - 添加了`BiorxivBanPolicy`类

### 修改文件

1. **`crawlers/crawler/settings.py`** - 启用了代理配置（记得提交前重新注释）

## 使用方法

### 基本命令

```bash
# 激活环境
conda activate torch2

# 进入爬虫目录
cd e:\codebase\uncheatable_eval\crawlers

# 运行爬虫（爬取所有分类）
scrapy crawl biorxiv \
    -a start_date=2025-12-18 \
    -a end_date=2026-01-01 \
    -a mineru_api="http://172.31.63.39:8999" \
    -a classification="all" \
    -o biorxiv_output.jsonl
```

### 参数说明

- `start_date`: 起始日期（格式：YYYY-MM-DD）
- `end_date`: 结束日期（格式：YYYY-MM-DD）
- `classification`: 分类（使用"all"爬取所有分类）
- `page_size`: 每页结果数（默认200）
- `mineru_api`: MinerU API地址
- `size_limit`: PDF大小限制（MB，默认50）

### 测试命令

```bash
# 小规模测试（仅爬取2篇）
scrapy crawl biorxiv \
    -s CLOSESPIDER_ITEMCOUNT=2 \
    -a start_date=2025-12-30 \
    -a end_date=2026-01-01 \
    -a page_size=10 \
    -a mineru_api="http://172.31.63.39:8999" \
    -o test_output.jsonl
```

## 输出格式

每篇论文的JSONL格式示例：

```json
{
  "content": "清理后的论文内容...",
  "category": "biorxiv_biorxiv",
  "date": "2025-12-30T00:00:00Z",
  "url": "https://www.biorxiv.org/content/...",
  "metadata": {
    "title": "论文标题",
    "raw_content": "原始OCR内容",
    "classification": "neuroscience"
  }
}
```

## 重要提醒

### 代理配置

爬虫需要使用代理访问BioRxiv（绕过Cloudflare保护）。

- **测试时**: 取消注释`settings.py`中的代理配置
- **提交前**: 重新注释代理配置，避免影响其他用户

```python
# settings.py中的配置
ROTATING_PROXY_LIST = ["http://127.0.0.1:8890"]  # 测试时取消注释
# ROTATING_PROXY_LIST = ["http://127.0.0.1:8890"]  # 提交前重新注释
```

## 实现细节

### 爬取流程

1. 生成BioRxiv搜索URL（基于日期范围）
2. 解析搜索结果页面，提取文章标题链接
3. 访问每篇文章的详情页
4. 从详情页提取PDF链接和学科分类
5. 检查PDF大小
6. 下载PDF并使用MinerU进行OCR
7. 清理文本并保存

### 分类策略

由于BioRxiv有27个学科分类，而项目约定subtitle必须为单个单词、小写、无非字母字符，因此采用：

- 统一使用`biorxiv`作为subtitle
- 在metadata中保存具体分类信息
- 类似arXiv的"other"分类处理方式

## 下一步

1. ✅ 代码已完成并推送到分支`feature/biorxiv-crawler`
2. ⏳ 使用代理进行完整测试
3. ⏳ 验证数据质量
4. ⏳ 合并到主分支前记得重新注释代理配置
