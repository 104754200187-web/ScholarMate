# LLM 服务与默认模型

本文件记录 ScholarMate Web 端内置模型服务的官方依据。核对日期：**2026-10-21**。
厂商会改名、下线或调整套餐；页面里的「模型」字段始终可编辑，默认值只作为起点。

| 服务 | 官方 Base URL | 鉴权方式 | 默认模型 | 官方文档 |
|---|---|---|---|---|
| DeepSeek | `https://api.deepseek.com` | `Authorization: Bearer` | `deepseek-flash` | [API 快速开始](https://api-docs.deepseek.com/) · [模型与价格](https://api-docs.deepseek.com/quick_start/pricing) |
| OpenAI | `https://api.openai.com/v1` | `Authorization: Bearer` | `gpt-6.1-sol` | [模型目录](https://platform.openai.com/docs/models) · [Responses API](https://developers.openai.com/api/docs/guides/text) |
| Anthropic | `https://api.anthropic.com` | `x-api-key` + `anthropic-version: 2023-06-01` | `claude-opus-5-5` | [首次调用](https://platform.claude.com/docs/en/get-started) · [模型总览](https://platform.claude.com/docs/en/about-claude/models/overview) |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta` | `x-goog-api-key` | `gemini-3.8-flash` | [API Key](https://ai.google.dev/gemini-api/docs/api-key) · [模型](https://ai.google.dev/gemini-api/docs/models) · [Interactions API](https://ai.google.dev/api/interactions-api) |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `Authorization: Bearer` | `qwen-plus` | [API Key](https://help.aliyun.com/zh/model-studio/get-api-key) · [模型列表](https://help.aliyun.com/zh/model-studio/models) |
| xAI | `https://api.x.ai/v1` | `Authorization: Bearer` | `grok-4.7` | [API 概览](https://docs.x.ai/docs/overview) · [模型](https://docs.x.ai/docs/models) |
| Mistral AI | `https://api.mistral.ai/v1` | `Authorization: Bearer` | `mistral-large-4` | [Chat API](https://docs.mistral.ai/api/endpoint/chat) · [模型总览](https://docs.mistral.ai/getting-started/models/models_overview/) |
| Moonshot Kimi | `https://api.moonshot.cn/v1` | `Authorization: Bearer` | `kimi-k3` | [Kimi API 快速开始](https://platform.moonshot.cn/docs/guide/start-using-kimi-api) |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `Authorization: Bearer` | `glm-5.3` | [OpenAI API 兼容](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction) · [模型概览](https://docs.bigmodel.cn/cn/guide/start/model-overview) |
| 字节豆包 Doubao | `https://ark.cn-beijing.volces.com/api/v3` | `Authorization: Bearer` | `doubao-seed-2-1-pro-260915` | [Base URL 与鉴权](https://www.volcengine.com/docs/82379/1298459) · [Chat API](https://ark.volcengine.com/region:cn-beijing/docs/chat-api) · [模型列表](https://www.volcengine.com/docs/82379/1330310) |
| 百度文心 ERNIE | `https://qianfan.baidubce.com/v2` | `Authorization: Bearer` | `ERNIE-5.1` | [OpenAI SDK 兼容](https://cloud.baidu.com/doc/qianfan/s/Hmh4suq26) · [模型与价格](https://cloud.baidu.com/doc/qianfan/s/wmh4sv6ya) · [API Key](https://cloud.baidu.com/doc/qianfan/s/wmh8l6tnf) |
| 腾讯混元 Hunyuan | `https://api.hunyuan.cloud.tencent.com/v1` | `Authorization: Bearer` | `hunyuan-turbos-latest` | [OpenAI 兼容接口](https://cloud.tencent.com/document/product/1729/111007) · [计费与模型](https://cloud.tencent.com/document/product/1729/97731) |
| MiniMax | `https://api.minimax.cn/v1` | `Authorization: Bearer` | `MiniMax-M3.1-Flash` | [OpenAI SDK 兼容](https://platform.minimaxi.com/docs/api-reference/text-openai-api) |
| Groq | `https://api.groq.com/openai/v1` | `Authorization: Bearer` | `openai/gpt-oss-120b` | [OpenAI 兼容](https://console.groq.com/docs/openai) · [模型列表](https://console.groq.com/docs/models) |
| OpenRouter | `https://openrouter.ai/api/v1` | `Authorization: Bearer` | `~openai/gpt-sol-latest` | [快速开始](https://openrouter.ai/docs/quickstart) · [模型目录](https://openrouter.ai/docs/guides/overview/models) |
| Cohere | `https://api.cohere.ai/compatibility/v1` | `Authorization: Bearer` | `command-a-plus-05-2026` | [OpenAI 兼容接口](https://docs.cohere.com/docs/compatibility-api) · [模型列表](https://docs.cohere.com/docs/models) |
| 硅基流动 SiliconFlow | `https://api.siliconflow.cn/v1` | `Authorization: Bearer` | `deepseek-ai/DeepSeek-V4-Flash` | [Chat API](https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions) · [快速开始](https://docs.siliconflow.cn/cn/userguide/quickstart) |
| 自定义 | 用户填写，仅允许 HTTPS 公网地址 | 默认 Bearer，可选 Responses API | 用户填写 | 由用户自行核对 |

说明：

- Anthropic 使用 Messages API；system / developer 消息由服务端合并为顶层 `system`。
- Gemini 使用官方当前推荐的 Interactions API，`store` 显式设为 `false`，不保留交互记录。
- 其余内置服务使用官方 OpenAI 兼容接口；内置服务的 Base URL 固定在服务端，浏览器只能覆盖模型名，不能把请求改道。
- 模型建议列表来自上表官方页面，但未锁定版本；上线新模型或旧模型下线后，用户可直接修改模型名，无需等待前端更新。
- 百度千帆使用 V2 推理服务；腾讯混元使用官方 OpenAI 兼容端点；MiniMax 使用国内站端点。国际站、Azure、AWS Bedrock、Vertex AI 等按部署/区域变化的服务请走「自定义」。

## 免费体验（免 Key）

Web 页面把厂商免费层标为「✅ 免费层」，把新用户赠送或试用额度标为
「🎁 新用户赠送/试用」。这些标记只说明厂商当前的计费口径，额度、限速和模型可用性
仍以厂商控制台为准。

没有 API Key 的用户可以从「设置自己的模型，或使用免费体验」弹窗里，使用管理员
在服务端配置的固定免费模型。浏览器只发送厂商与模型名，服务端按 `llm.FREE_ACCESS`
做 allow-list 校验并用服务端 Key 发起请求；Key 不会出现在响应、任务表或历史快照中。
未配置 Key 的厂商不会出现在可选列表里，也不会回退到 DeepSeek/OpenAI 的付费 Key。

| 厂商 | 页面标注的免费/试用模型 | 官方依据 |
|---|---|---|
| Google Gemini | `gemini-3.8-flash`、`gemini-3.5-flash-lite`（免费层） | [Gemini 定价](https://ai.google.dev/gemini-api/docs/pricing) |
| Groq | `openai/gpt-oss-120b`、`openai/gpt-oss-20b`、`qwen/qwen3.8-27b`；Free Plan 摘要约 30 RPM / 1K RPD / 8K TPM | [Groq 限速](https://console.groq.com/docs/rate-limits) · [模型](https://console.groq.com/docs/models) |
| 智谱 GLM | `GLM-4.7-Flash`、`GLM-4.5-Flash`、`GLM-4-Flash-250414`（免费文本模型） | [智谱定价](https://bigmodel.cn/pricing) |
| OpenRouter | `minimax/minimax-m3.1-flash:free`、`google/gemma-4-26b-a4b-it:free`、`google/gemma-4-31b-it:free`；以 `/api/v1/models` 的零价条目为准 | [模型目录](https://openrouter.ai/models) · [API](https://openrouter.ai/docs/quickstart) |
| Cohere | `command-a-plus-05-2026`、`command-a-03-2025`、`command-a-reasoning-08-2025`；Trial Key 官方标注约 1000 次 API 调用/月 | [试用 Key 与限速](https://docs.cohere.com/docs/rate-limits) |
| 百度文心 ERNIE | `ERNIE-4.5-Turbo-128K`、`ERNIE-4.5-Turbo-32K`；新用户各 100 万 tokens，有效期 3 个月 | [新用户免费额度](https://cloud.baidu.com/doc/qianfan/s/1mh4stp3t) |
| 腾讯混元 Hunyuan | `hunyuan-a13b`；首次开通的一次性免费资源包共 100 万 tokens，有效期 1 年 | [计费与免费额度](https://cloud.tencent.com/document/product/1729/97731) |

服务端免 Key 通道当前每个厂商只开放一个**固定**模型（见 `llm.FREE_ACCESS`），
不能通过修改请求模型去使用同一 Key 上的付费模型。启用方式是配置
`GEMINI_API_KEY` / `GROQ_API_KEY` / `ZHIPU_API_KEY` / `OPENROUTER_API_KEY` /
`COHERE_API_KEY` / `BAIDU_API_KEY` / `HUNYUAN_API_KEY` 中的一个或多个；
没有配置的厂商只显示为不可用。

保护措施：

- 每个浏览器每天默认最多 `5` 次免费体验（`FREE_LLM_DAILY_REQUESTS=0` 关闭该通道）。
- 每个进程每天默认最多 `50` 次（`FREE_LLM_GLOBAL_DAILY_REQUESTS=0` 取消全站上限）。
- 限额计数在内存中，重启会归零；它是防滥用护栏，不是计费或审计系统。
- 免费模型通常有更低的 RPM/TPM；任务可能因限流失败，用户可改用自带 Key 重试。
- OpenRouter 零价模型和 Gemini 免费层可能调整或下线；上线前应再次核对厂商页面。
