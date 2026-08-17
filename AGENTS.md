# AGENTS.md

这是 GhostAP 中 AI 编码代理的简明指南。保持简洁：此文件是启动上下文，而非项目文档。只有在规则能防止已知失败或帮助代理更快找到正确工具时才添加规则。

## 项目概述

GhostAP 是一个飞书/Lark 机器人服务，用于通过出站 WebSocket 连接进行安全的远程 shell 执行和 AI 辅助开发。用户可以通过聊天运行 shell 命令、管理项目，并驱动 Coco、Claude、Aiden、Codex、Gemini、Traex、Grok 和 DSH 等编程工具。

## 命令

仅使用 `uv`；本仓库中绝不使用 pip/conda。

```bash
uv sync --group dev
uv run python -m src.main
uv run python -m src.main --validate
uv run python -m pytest tests/ -q -m "not slow"
uv run python -m pytest tests/ -q
uv run python -m pytest tests/test_acp_client.py -q
uv run python scripts/test_inventory.py tests/
```

进行针对性修改时，先运行最相关的测试，然后在修改涉及共享路由、卡片渲染、锁、配置或会话代码时扩大测试范围。测试准入、删减与快慢分层规则见 `docs/testing.md`。

## 工作规则

- 修改行为前阅读 `.Memory/Abstract.md`；这是项目本地的近期决策和陷阱索引。
- 在编辑已建立的模块前，使用 `rg` 检查现有模式。
- 调用 Codex `spawn_agent` 时，显式传 `agent_type` 必须同时使用 `fork_turns="none"` 或正整数；全历史 fork 必须省略 `agent_type`，否则子代理会在启动前被拒绝。
- 保持修改范围明确。修复局部问题时不要重构无关代码。
- 所有功能和 bug 修复都需要测试。对于涉及的合约，优先使用针对性回归测试。
- GhostAP 至少支持 Linux 与 macOS。涉及进程、sandbox、文件锁、路径/临时目录、权限位、环境变量或平台工具时，必须用平台分支或能力探测保持两端契约并补回归；不得把 Linux-only 假设带到 macOS，也不得以放宽 fail-closed 安全边界换取兼容性。
- 所有测试失败、异常退出或缺失最终摘要都必须查明原因并修复；禁止跳过、掩盖或当作通过。若根因是测试本身错误，需谨慎修正测试及其契约并保留回归证据。
- 完成判定必须回读用户原始验收范围；若最终总结仍列出未实现的用户要求、未验证的核心路径或相关失败测试，不得标记完成，应继续执行或明确报告阻塞/失败。
- 每个独立 bug 修复或需求完成并验证后，立即创建一个聚焦 commit 作为回滚点；不同问题分开提交。每次修改提交后必须立即 push 到当前所在分支；push 失败必须查明原因并明确报告，未成功 push 不得视为完成。
- 机密和环境特定值通过 `src/config/` 从 `.env` 获取。切勿硬编码凭据或令牌。
- 测试、探测和临时辅助工具应放在 `tests/`、`scripts/`、`ux/` 或 `/tmp` 下；保持仓库根目录整洁。
- 完成有意义的任务后，用详细条目更新 `.Memory/{YYYY-MM-DD}.md`：更改内容、原因、验证和任何后续风险。同时在 `.Memory/Abstract.md` 中添加一行摘要（约20个字符）和日期引用，以便读者在每日文件中找到完整记录。
- 中/低审计发现放入 `.Memory/Backlog.md`；高正确性或安全性发现应立即修复。修复后移除待办项。
- 提交消息必须遵循 `docs/commit-message-guidelines.md`。

## 指南原则

将此文件用作指南，而非维基百科：

- 将持久规则放在这里；将历史和证据放在 `.Memory/` 中。
- 优先选择特定的失败衍生规则而非通用建议。
- 如果规则可以通过测试、钩子或类型化 API 强制执行，就在那里强制执行，并在此处只保留简短指针。
- 当代码库或工具不再需要时，删除过时规则。
- 将 Coco/Claude/Aiden/Codex/Gemini/Traex/Grok/DSH 视为 GhostAP 编程抽象背后的工具后端。除非传输或功能确实不同，否则避免添加后端特定分支。
- 机器人管理员引导是单向的：仅当 `ADMIN_USER_IDS` 为空时，才接受任意用户在主 Bot 私聊中发送 `/setadmin`；群聊中的首次设置一律拒绝。之后只有配置的管理员可以替换 `.env` 中的单个管理员。

## 架构指针

从这些模块开始，而不是阅读整个树：

- `src/feishu/ws_client.py` 和 `src/feishu/dispatcher.py`：WebSocket 入口、消息路由和交互模式调度。
- `src/feishu/handlers/`：命令处理器。使用 `BaseHandler` 消息辅助函数：`reply_text`、`reply_card`、`update_card`、`send_card_to_chat`、`send_text_to_chat`。
- `src/mode/`：`InteractionMode` 和每聊天/项目模式状态。
- `src/acp/`：ACP 会话、提供者、诊断和支持 ACP 的编程工具的模型/工具发现。
- `src/deep_engine/`、`src/spec_engine/`、`src/workflow_engine/`：长时间运行的执行策略。
  - `src/workflow_engine/`：JS 编排的多代理并行执行。关键模块：`commands.py`（SSOT 命令集）、`engine.py`（桥接 + 运行时）、`executor.py`（每代理调用会话）、`tool_registry.py`（动态发现）、`script_gen.py`（提示模板 + 验证）、`renderer.py`（飞书卡片）。需要 Node.js >= `NODE_MIN_VERSION`（在 `src/workflow_engine/constants.py` 中定义）；所有面向用户的"Node 版本过旧"消息都源自此常量。
- `src/card/`：飞书卡片构建器、渲染管道、会话编排和交付。
- `src/project/`、`src/project_chat/`、`src/thread/`：项目上下文、项目聊天绑定和线程上下文。
- `src/chat_lock.py`、`src/repo_lock.py`、`src/utils/lock_order.py`：聊天/仓库锁定和锁定顺序强制执行。
- `src/config/`：pydantic 设置包和配置验证。

## 自主工作系统 (`src/autonomous/`)

- **现役生产面是 Employee Department**。旧 Goal/Run Manager、AgentRuntime、Policy/Broker、Scheduler/Reporter/Verifier 已退役，不要重新接回生产入口。

- **Journal 是唯一事实源**。所有状态变更通过 `JournalWriter.write_event()` 记录。投影和快照可从 Journal 重放重建。
- **域对象冻结**。`domain/` 下的 dataclass 都是 `frozen=True`，状态变更使用 `dataclasses.replace()` 而非赋值。
- **Effect 在派发前必须锚定**。PREPARED 和 EXECUTING 帧必须 fsync 并锚定后才能发起外部调用。
- **终态需要 finalization**。Run 在有未解决 Effect 或未处置已提交 Effect 时不能进入终态。
- **默认拒绝**。`src/trust/action_matrix.py` 对所有操作默认拒绝，需显式授权。
- **Assist 不写入**。`assist` 模式下系统只读，R4 风险始终拒绝。
- **飞书 SDK 使用官方包**。消息/卡片投递使用 `lark-oapi`，WebSocket 事件订阅使用 `lark-channel-sdk`。不要手写 HTTP 调用。

关键模块入口：

- `src/autonomous/provisioning/composition.py`：Employee Department 生产组装根。
- `src/autonomous/provisioning/hire_service.py`：员工创建与配置编排。
- `src/autonomous/gateway/coordinator.py`：现役 Effect 派发与恢复协调。
- `src/autonomous/domain/state_machine.py`：纯状态转换函数（`transition_run/plan/step/effect`）。
- `src/autonomous/journal/writer.py`：单写入者，fsync + flock。
- `src/trust/action_matrix.py`：权威动作授权矩阵。
- `src/autonomous/runtime/employee_actor.py`：现役员工 actor 执行循环。
- `src/autonomous/supervisor/employee_channels.py`：员工 Channel SDK 进程生命周期。
- `src/autonomous/provisioning/channel_worker.py`：由 supervisor 通过文件路径启动的 Channel 子进程入口。

测试命令：

```bash
uv run pytest tests/autonomous/ -q
uv run ruff check src/autonomous/           # 0 错误
```

## 策略与传输

GhostAP 有两个独立的维度：

- 执行策略：普通编程、Deep、Spec、Workflow 和 Autonomous。
- 工具传输：ACP 直接模式和 shell CLI 桥接模式。

保持这些维度分离。新的编程功能通常应在 Coco、Claude、Aiden、Codex、Gemini、Traex 和 Grok 上工作，除非用户明确限定范围或后端不支持。

状态范围也是产品合约：

- SMART 是默认聊天/项目状态，可直接路由简单意图或类 shell 命令。
- 普通工具入口如 `/coco`、`/codex`、`/aiden`、`/claude`、`/gemini`、`/traex` 和 `/grok` 设置持久聊天+项目编程状态，直到 `/exit`。
- Deep、Spec 和 Workflow 是作用于飞书话题/根线程的引擎策略；它们不得替换聊天+项目编程状态。Autonomous 同理。
- SMART 中的类 shell 文本必须保持 shell 执行，包括 `./restart.sh rr` 等命令，而不是被项目聊天自由文本编程路由窃取。

## 卡片与 UI 规则

- 普通编程卡片遵循一个用户任务一张主卡。在顶部显示总体任务列表、当前活动任务和
  子代理摘要；子代理文本、工具调用与图片进入主卡执行流，不再创建独立飞书消息。
- 只有卡片超过飞书容量限制时才创建续接卡，并保留先前卡片内容。
- 避免空工具/详情块。
- 对于 UI 设计更改，在实现前在 `ux/` 下创建或更新 HTML 预览，然后使生产代码与已审查的预览对齐。
- 尊重卡片分层：处理器使用会话/协议 API；会话编排渲染和交付；渲染保持纯净；交付不导入会话。

## 导入边界

卡片管道有严格的单向依赖方向：

```text
handler -> session -> render
                  -> delivery
```

- `render` 不得导入 `delivery`。
- `delivery` 不得导入 `session`。
- 处理器应依赖协议和外观，而非具体渲染器内部。
- 跨层共享类型应放在 `src/card/protocols.py` 或 `src/card/events/` 中。
- 仅在保持此方向时使用 `TYPE_CHECKING` 或局部延迟导入。

## 当前注意事项

- `CardBuilder.build_engine_card()` 已移除。静态卡片使用 `build_info_card()`；引擎/进度卡片通过 `CardSession` 管道。
- Spec 通过 `SpecManager.persist_result` 持久化上下文；Deep 使用 `ContextPersistenceHook`。
- `ACPSessionManager` 负责会话密钥解析和锁定。不要在业务代码中手动解析会话密钥。
- 飞书卡片 JSON 严格。如果 `logs.log` 中出现模式错误，修复发出的结构并在构建器或渲染器周围添加回归测试。
- 对于重启/启动问题，在更改应用代码前检查 `logs.log` 和 `[RESTART]` 标记；将脚本延迟与 Python 冷启动分开。

## Workflow 模式 (`/wf`)

`WorkflowHandler` 负责 `/wf` 命令，允许用户用自然语言描述多步骤任务，由编排 Agent 生成并执行 Node.js 工作流脚本。**三步流程**如下：

1. **① 选择主编排Agent** — 选择一个工具+模型组合来驱动脚本生成。组合卡片允许展开工具查看其模型面板，或直接点击 "+ 添加 <工具>" 使用默认模型。此处不需要多选：编排器是单个选择的 Agent。
2. **② 选择评审Agent** — 使用相同的组合卡片界面。可以选择一个或多个工具+模型组合作为独立评审者，或点击 **Auto** 快捷按钮跳过独立评审，由编排器进行自我评审。跳过评审适用于低风险变更，可避免额外的 Agent 调用成本。
3. **③ 确认并执行** — 当两步都非空（或步骤2启用了Auto）后，引擎通过 `src/workflow_engine/script_gen.py` 构建 JS 工作流，验证输出（元数据导出、括号平衡、至少一个 `agent()`/`workflow()`/模式原语调用、无禁止的 `require('fs'|child_process|net|...)` 逃逸），显示确认卡片列出阶段、工具和简短预览，用户确认后执行脚本。进度通过 `WorkflowProgressRenderer` 流式传输。

### Dynamic Workflow 编排模式

Workflow 引擎提供 6+2 个高阶编排原语，作为 JS 运行时全局函数（`src/workflow_engine/runtime/runtime.js`）：

| 原语 | 模式 | 用途 |
| --- | --- | --- |
| `classify(input, categories, opts)` | Classify-and-Act | 先分类后路由到不同处理逻辑 |
| `fanout(input, workers, opts)` | Fan-out-and-Synthesize | 拆分并行执行后合成 |
| `verify(output, opts)` | Adversarial Verification | 对抗性验证+循环修订 |
| `generate(count, generatorFn, filterFn, opts)` | Generate-and-Filter | 生成多方案后过滤排序 |
| `tournament(contestants, judgeFn, opts)` | Tournament | 淘汰赛决出最佳方案 |
| `loop(taskFn, opts)` | Loop-Until-Done | 循环执行直到收敛/停止条件 |
| `sequence(steps)` | Sequential | 严格顺序执行（每步传递结果） |
| `race(contestants, opts)` | First-to-Finish | 竞速取第一个有效结果 |

**比例原则**：简单任务用 1 个 agent() 调用；中等任务用 fanout/sequence（3-5 calls）；复杂任务才组合多个模式。

**安全约束**：`generate()` 上限 50；`loop()` 硬上限 50；`MAX_TOTAL_AGENTS`（200）由 Python 侧强制。所有原语通过 `sandboxWrapHostFn` 包装。


错误处理：
- 任一步骤为空选择时，卡片中会显示内联错误；用户需要选择至少一个工具/模型并重试。
- 验证失败的脚本会被拒绝，并返回结构化错误列表（缺少元数据、不安全模式等）—— 用户从确认卡片重新生成。
- 运行中的工作流会阻止新的 `/wf` 调用，必须使用 `/stop_wf` 或进度卡片上的取消按钮停止。

### 快速开始

#### 命令速查

| 命令 | 用途 |
| --- | --- |
| `/wf <需求描述>` | 从需求描述启动新工作流 |
| `/stop_wf` | 中止当前运行的工作流 |
| `/wf_status` | 显示活动工作流进度和已选工具 |
| `/wf_help` | 聊天内帮助文本 |

#### 交互流程

1. **输入命令**：在飞书聊天中输入 `/wf <您的需求>`，例如 `/wf 帮我创建一个用户登录页面`
2. **选择主编排Agent**：在弹出的卡片中选择一个工具+模型组合作为编排器
3. **选择评审Agent**：选择一个或多个评审工具+模型组合，或点击 **Auto** 按钮跳过评审
4. **确认执行**：查看生成的脚本预览后点击确认按钮开始执行
5. **查看进度**：实时查看工作流执行进度

#### 取消/回退操作

- 在确认阶段点击 **取消** 按钮取消工作流
- 在执行阶段使用 `/stop_wf` 命令或点击进度卡片上的取消按钮停止工作流
- 如果遇到错误，卡片会显示错误提示和处理建议

#### 三步工作流流程

工作流使用组合卡片界面完成完整的三步流程：

1. **① 编排器步骤（步骤1）**：选择恰好一个工具+模型组合来生成工作流脚本。使用顶部的步骤指示器跟踪进度（当前=1/3）。

2. **② 评审步骤（步骤2）**：选择一个或多个工具+模型组合来评审生成的脚本，或使用 **Auto** 按钮跳过独立评审。步骤指示器显示当前=2/3。

3. **③ 确认步骤（步骤3）**：查看生成的工作流脚本并确认执行。步骤指示器显示当前=3/3。

#### 组合卡片功能
- **工具+模型内联展开**：点击任意工具可内联展开并查看其可用模型，无需导航到单独卡片。
- **步骤指示器**：显示当前步骤（1/3、2/3 或 3/3）和整体进度。
- **Auto 选项**：在评审步骤中，跳过独立评审并使用编排器 Agent 进行自我评审。
- **移除/清除按钮**：单击即可移除单个选择或清除所有选择。
- **空选择验证**：通过显示内联错误消息防止空选择继续。

#### 跳过评审

在评审步骤中使用 **Auto** 按钮跳过独立评审的场景：
- 进行低风险变更（例如，小的 bug 修复、文档更新）
- 处于快速原型开发模式
- 信任编排器 Agent 的自我评审能力

**风险提示**：
- 跳过独立评审可能会遗漏潜在问题
- 建议在生产环境或高风险变更时启用独立评审
- Auto 模式下，角色由 LLM 动态分配

**重新启用评审**：
- 如果在步骤2选择了 Auto，可以返回到步骤2重新选择评审工具
- 在确认卡片上可以看到评审状态（Auto 或具体评审工具）
- 如果需要，可以点击"重新选择"按钮返回工具选择界面

#### 脚本生成与确认
- **动态角色分配**：角色（编排器/评审者）由 LLM 从任务描述中动态推断，而非由用户静态选择。
- **脚本预览**：确认卡片显示生成的工作流脚本预览，包含关键细节：
  - 编排器工具/模型
  - 评审工具/模型（如果跳过评审则显示 "Auto"）
  - 阶段分解
- **执行控制**：确认执行脚本，或在需要更改时重新生成。

#### Agent() 调用执行

当工作流运行 `agent()` 调用时：
- 每个 Agent 调用使用选定的工具/模型组合
- 评审 Agent 对编排器的工作提供反馈
- 最终输出将所有 Agent 结果合并为一个连贯的交付物
- 进度通过工作流进度卡片实时流式传输
## 全自动执行契约

普通编程、Deep、Spec、Workflow 在收到任务后均应自动推进到成功或明确失败终态：

- 使用项目已保存配置；缺失时采用可用的推荐工具和后端默认模型。
- 普通、安全、可逆的选择自动采用推荐项；高风险且未经原始请求精确授权的操作自动拒绝或跳过，并继续安全部分。
- Agent 提问、Review 不确定、格式修复和暂时性失败使用有界自动恢复；耗尽后明确失败，不进入等待用户状态。
- 只保留用户主动的停止/取消和显式配置入口；任务主路径不得要求选择 Agent、确认脚本、批准继续或手动恢复。
- ACP 权限保持 fail-closed，只允许明确的一次性安全授权；禁止以跳过权限检查换取自动化。

Workflow 使用 `src/workflow_engine/runtime/runtime.js` 提供的 `classify`、`fanout`、`verify`、`generate`、`tournament`、`loop`、`sequence`、`race` 原语动态生成任务专用 JS。运行时负责确定性控制流，Agent 负责语义工作；简单任务保持单 Agent，复杂度增长时再组合原语。

Workflow 飞书卡片显示任务摘要、阶段统计和所有直接 `agent()` 调用的调度状态；每个 Agent 只显示一条最新操作，终态通过分页或附件完整交付结果。ACP 内部嵌套 Agent 在协议缺少权威列表时只能标记为观测信息，不得从陈旧快照推断终态。
## Workflow 模式 (`/wf`)

`WorkflowHandler` 负责自然语言 Workflow。当前交互和执行契约如下：

1. `/wf <需求>` 先显示 owner-bound Agent Pool 选择卡；只有发起者可在同一 chat/project/session 中修改。
2. 池必须包含 1–8 个 `tool+model` Agent，并使用稳定 `A1`、`A2` 标识。同一 tool 可选择不同 model；完全相同的 tool/model 组合拒绝。
3. 编排器默认由系统从已确认池中择优，也可指定池内成员。
4. 用户只确认一次 **“使用此池开始编排”**。确认后池和编排器冻结，脚本生成、验证与执行全自动完成，不再显示脚本确认或继续批准。

脚本生成和运行遵循同一池边界：

- Prompt、fallback 和 `WorkflowRunSpec` 只使用已确认池；不得吸收 registry 中未选择的工具。
- 每个直接或动态 Agent 调用都用 `agentId` 引用池内成员；tool/model 由冻结绑定决定，脚本值不能覆盖。
- `meta.agentPlan` 用 `agentId` 表示静态节点，用 `runtime: true` 与 `candidateAgentIds` 表示运行时候选。
- 进度卡展示编排器、完整池、静态计划、动态候选和实际 `A-id/tool/model/current operation/result`；实际运行绑定是权威事实。
- 运行时原语为 `classify`、`fanout`、`verify`、`generate`、`tournament`、`loop`、`sequence`、`race`。简单任务保持少量调用；`generate`、`loop` 上限 50，`MAX_TOTAL_AGENTS` 上限 200。

生成和终态规则：

- 生成期间或执行中再次发送 `/wf` 会被拒绝；使用 `/stop_wf` 停止，或等待当前 Workflow 终态。
- 脚本生成使用 120 秒 activity-idle timeout，并共享 600 秒 hard deadline；池内 fallback 每个唯一成员最多尝试一次，切换成员不重置 hard deadline。
- 生成、验证或执行耗尽后进入 `FAILED`；用户停止进入 `CANCELLED`。两者都是明确终态，不进入等待确认。

命令速查：

| 命令 | 用途 |
| --- | --- |
| `/wf <需求描述>` | 选择 Agent Pool 并启动 Workflow |
| `/stop_wf` | 中止当前 Workflow |
| `/wf_status` | 查看池、计划和执行进度 |
| `/wf_help` | 查看 Workflow 帮助 |

## 全自动执行契约

普通编程、Deep、Spec 在收到任务后自动推进；Workflow 在 Agent Pool 单次确认后自动推进。所有模式都必须到达成功或明确失败终态：

- 使用项目已保存配置；缺失时采用可用的推荐工具和后端默认模型。
- 普通、安全、可逆的选择采用推荐项；高风险且未经原始请求精确授权的操作拒绝或跳过，并继续安全部分。
- Agent 提问、Review 不确定、格式修复和暂时性失败使用有界自动恢复；耗尽后明确失败。
- 除 Workflow 的单次池确认外，不增加 Agent 选择、脚本确认、继续批准或手动恢复门。
- ACP 权限保持 fail-closed，只允许明确的一次性安全授权。
