"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity, AlertTriangle, BookOpen, Bot, Boxes, CheckCircle2, ChevronDown,
  CircleDollarSign, ClipboardCheck, Cloud, Code2, Database, Download, FileCheck2, FileText, GitBranch,
  HelpCircle, LayoutDashboard, LockKeyhole, Menu, Microscope, Moon, Network,
  Search, Settings, ShieldCheck, Sun, Upload, X, XCircle,
} from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Toaster, toast } from "sonner";

type View = "overview" | "experiments" | "evidence" | "protocols" | "tasks" | "compute" | "datasets" | "models" | "literature" | "repository" | "reports" | "audit";

const runs = [
  { id: "A14", dataset: "CITICSEC", model: "iTransformer", seed: 2021, status: "校准通过", tone: "success", duration: "14.9 秒", integrity: "已验证", mse: "0.3336" },
  { id: "A13", dataset: "CITICSEC", model: "PatchTST", seed: 2021, status: "校准通过", tone: "success", duration: "43.4 秒", integrity: "已验证", mse: "0.2583" },
  { id: "A12", dataset: "CITICSEC", model: "DeReFusion", seed: 2021, status: "校准通过", tone: "success", duration: "71.1 秒", integrity: "已验证", mse: "0.2568" },
  { id: "A11", dataset: "CITICSEC", model: "DLinear", seed: 2021, status: "校准通过", tone: "success", duration: "2.7 秒", integrity: "已验证", mse: "0.4292" },
  { id: "A10", dataset: "BONDETF", model: "TimesNet", seed: 2021, status: "模型族暂停", tone: "neutral", duration: "—", integrity: "未运行", mse: "—" },
  { id: "A05", dataset: "BOND10Y", model: "TimesNet", seed: 2021, status: "资源阻塞", tone: "danger", duration: "205 秒", integrity: "部分", mse: "—" },
];

const nav = [
  { id: "overview", label: "项目总览", icon: LayoutDashboard },
  { id: "experiments", label: "实验注册表", icon: Microscope, count: "15" },
  { id: "protocols", label: "协议与门禁", icon: LockKeyhole, count: "1" },
  { id: "evidence", label: "主张与证据", icon: Network, count: "4" },
  { id: "tasks", label: "任务与 Agent", icon: Bot, count: "3" },
  { id: "compute", label: "算力与预算", icon: CircleDollarSign },
] as const;

const secondary = [
  { id: "datasets", label: "数据集", icon: Database }, { id: "models", label: "模型注册表", icon: Boxes },
  { id: "literature", label: "文献知识库", icon: BookOpen }, { id: "repository", label: "代码仓库", icon: GitBranch },
  { id: "reports", label: "报告中心", icon: FileText }, { id: "audit", label: "审计日志", icon: ShieldCheck },
] as const;

const allNavigation = [...nav, ...secondary];

function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: string }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function MetricCard({ label, value, note, accent }: { label: string; value: string; note: string; accent?: string }) {
  return <article className="metric-card"><div className="metric-label">{label}</div><div className="metric-value" style={{ color: accent }}>{value}</div><div className="metric-note">{note}</div></article>;
}

function Roadmap() {
  const items = [["Gate A", "失败", "failed"], ["F1", "窄成功", "partial"], ["C1", "未支持", "failed"], ["Stage A", "待审计", "active"], ["Stage B", "已锁定", "locked"], ["Stage C", "未开始", "idle"], ["Stage D", "未开始", "idle"]];
  return <div className="roadmap" aria-label="研究路线图">{items.map(([name, status, tone], index) => <div className="roadmap-segment" key={name}><div className={`roadmap-node ${tone}`}><span className="roadmap-dot" /><strong>{name}</strong><small>{status}</small></div>{index < items.length - 1 && <div className="roadmap-line" />}</div>)}</div>;
}

function RunsTable({ compact = false }: { compact?: boolean }) {
  const [filter, setFilter] = useState("全部状态");
  const visible = useMemo(() => filter === "全部状态" ? runs : runs.filter((r) => r.status === filter), [filter]);
  return <div className="table-shell">
    {!compact && <div className="table-toolbar"><div className="filter-group">{["全部状态", "校准通过", "资源阻塞", "模型族暂停"].map((item) => <button className={filter === item ? "filter active" : "filter"} onClick={() => setFilter(item)} key={item}>{item}</button>)}</div><button className="outline-button">导出清单</button></div>}
    <Table><TableHeader><TableRow><TableHead>运行</TableHead><TableHead>数据集</TableHead><TableHead>模型</TableHead><TableHead>状态</TableHead>{!compact && <TableHead>Seed</TableHead>}<TableHead>耗时</TableHead>{!compact && <TableHead>MSE</TableHead>}<TableHead>完整性</TableHead></TableRow></TableHeader>
      <TableBody>{visible.slice(0, compact ? 5 : visible.length).map((run) => <TableRow key={run.id}><TableCell className="run-id">{run.id}</TableCell><TableCell>{run.dataset}</TableCell><TableCell>{run.model}</TableCell><TableCell><Badge tone={run.tone}>{run.status}</Badge></TableCell>{!compact && <TableCell className="mono">{run.seed}</TableCell>}<TableCell>{run.duration}</TableCell>{!compact && <TableCell className="mono">{run.mse}</TableCell>}<TableCell><span className={run.integrity === "已验证" ? "verified" : "muted"}>{run.integrity === "已验证" && <CheckCircle2 size={14} />} {run.integrity}</span></TableCell></TableRow>)}</TableBody>
    </Table>
  </div>;
}

function Overview() {
  return <>
    <section className="decision-banner"><div className="decision-icon"><LockKeyhole size={20} /></div><div><strong>Stage B 暂不可启动</strong><p>先解决 TimesNet 校准缺口，并完成 Stage A 产物审计与协议决策。</p></div><button>查看阻塞原因</button></section>
    <section className="metrics-grid"><MetricCard label="Stage A 运行" value="12 / 15" note="80% · 三项待处理" accent="#2457d6" /><MetricCard label="证据成熟度" value="早期" note="1 项部分支持 · 2 项失败" accent="#b7791f" /><MetricCard label="当前算力" value="0" note="没有运行中的计算任务" accent="#687386" /><MetricCard label="预计云端费用" value="¥25–126" note="完整 Phase 1 粗略区间" accent="#16845b" /></section>
    <section className="panel roadmap-panel"><div className="panel-header"><div><span className="eyebrow">RESEARCH FLOW</span><h2>研究路线与门禁</h2></div><Badge tone="frozen">协议 v1.1 已冻结</Badge></div><Roadmap /></section>
    <div className="two-column"><section className="panel"><div className="panel-header"><div><span className="eyebrow">RECENT RUNS</span><h2>最近实验</h2></div><button className="text-button">查看全部 →</button></div><RunsTable compact /></section>
      <section className="panel attention-panel"><div className="panel-header"><div><span className="eyebrow">ATTENTION</span><h2>需要决策</h2></div><Badge tone="warning">2 项</Badge></div><div className="attention-item"><AlertTriangle size={18} /><div><strong>TimesNet 资源阻塞</strong><p>CPU 环境无法在时限内完成首轮。需要批准 GPU 补跑或协议修订。</p><button>打开决策记录</button></div></div><div className="attention-item"><GitBranch size={18} /><div><strong>本地领先远程 2 个提交</strong><p>八个校准结果目录尚未提交；此前远程连接被重置。</p><button>查看仓库状态</button></div></div></section></div>
    <section className="panel claims-panel"><div className="panel-header"><div><span className="eyebrow">CLAIMS</span><h2>核心主张状态</h2></div><button className="text-button">证据图谱 →</button></div><div className="claim-row"><XCircle className="claim-icon fail" /><div><strong>DeReFusion 普遍优于简单模型</strong><p>Gate A 存在容量混杂，C1 未支持预设关系。</p></div><Badge tone="danger">证据不足</Badge></div><div className="claim-row"><Activity className="claim-icon partial" /><div><strong>容量受控条件下存在局部结构信号</strong><p>F1 提供窄范围支持，尚不能泛化。</p></div><Badge tone="warning">部分支持</Badge></div><div className="claim-row"><HelpCircle className="claim-icon idle" /><div><strong>复杂度收益取决于市场状态与预测长度</strong><p>等待 Phase 1 的筛选、确认与滚动验证。</p></div><Badge tone="neutral">尚未检验</Badge></div></section>
  </>;
}

function Experiments() { return <section className="panel full-panel"><div className="panel-header"><div><span className="eyebrow">REGISTRY</span><h2>Stage A 实验注册表</h2><p className="section-copy">所有指标仅用于执行校准，不构成模型排名或论文结论。</p></div><button className="primary-button">导入运行回执</button></div><RunsTable /></section>; }

function Protocols() {
  const checks: [string, boolean][] = [["数据集哈希与冻结配置一致", true], ["训练、验证、测试切分已审计", true], ["12 个成功运行产物完整", true], ["TimesNet 三个校准任务完成", false], ["Stage A 总结通过独立审查", false], ["Stage B 运行授权", false]];
  return <div className="detail-grid"><section className="panel"><div className="panel-header"><div><span className="eyebrow">PROTOCOL</span><h2>Phase 1 · v1.1</h2></div><Badge tone="frozen">冻结</Badge></div><div className="protocol-meta"><div><span>冻结时间</span><strong>2026-09-19</strong></div><div><span>主要问题</span><strong>复杂度何时产生可靠收益？</strong></div><div><span>运行规模</span><strong>819 项（A–D）</strong></div><div><span>结果可见性</span><strong>Stage A 已揭示</strong></div></div><button className="outline-button wide">查看完整预注册文档</button></section><section className="panel"><div className="panel-header"><div><span className="eyebrow">GATE</span><h2>Stage B 启动检查</h2></div><Badge tone="danger">阻塞</Badge></div><div className="check-list">{checks.map(([label, done]) => <div className="check-item" key={label}>{done ? <CheckCircle2 className="ok" /> : <XCircle className="no" />}<span>{label}</span></div>)}</div></section></div>;
}

function Evidence() {
  const claims = [{ title: "DeReFusion 普遍优于简单模型", status: "不支持", tone: "danger", level: "低", basis: "Gate A 容量混杂；C1 失败", publish: "不可" }, { title: "容量受控条件下存在局部结构信号", status: "部分支持", tone: "warning", level: "早期", basis: "F1 窄范围成功", publish: "仅限局限性表述" }, { title: "复杂度收益依赖市场状态与预测长度", status: "未检验", tone: "neutral", level: "无", basis: "等待 Phase 1 B–D", publish: "不可" }, { title: "NS 机制可以解释收益边界", status: "未检验", tone: "neutral", level: "无", basis: "尚未预注册", publish: "不可" }];
  return <section className="panel full-panel"><div className="panel-header"><div><span className="eyebrow">EVIDENCE LEDGER</span><h2>主张与证据</h2><p className="section-copy">每条陈述必须能追溯到协议、实验、分析和原始产物。</p></div><button className="outline-button">新建主张</button></div><div className="evidence-list">{claims.map((c) => <article className="evidence-card" key={c.title}><div className="evidence-title"><h3>{c.title}</h3><Badge tone={c.tone}>{c.status}</Badge></div><dl><div><dt>证据等级</dt><dd>{c.level}</dd></div><div><dt>主要依据</dt><dd>{c.basis}</dd></div><div><dt>发表资格</dt><dd>{c.publish}</dd></div></dl><button className="text-button">查看证据链 →</button></article>)}</div></section>;
}

function Tasks() { return <div className="kanban"><section><div className="kanban-heading"><span>进行中</span><Badge tone="info">1</Badge></div><article className="task-card"><div className="task-top"><span className="mono">T-P1A-REVIEW</span><Badge tone="info">审计</Badge></div><h3>完成 Stage A 产物审计与总结</h3><p>校验哈希、补充 A10/A15 暂停记录，并形成阶段裁定。</p><div className="task-footer"><span className="avatar">C</span><span>Codex</span><span className="task-lock"><LockKeyhole size={13} /> 已锁定</span></div></article></section><section><div className="kanban-heading"><span>等待决策</span><Badge tone="warning">1</Badge></div><article className="task-card"><div className="task-top"><span className="mono">T-GPU-001</span><Badge tone="warning">审批</Badge></div><h3>TimesNet GPU 补跑方案</h3><p>先用单张 RTX 4090 完成三个校准任务，预算上限 ¥50。</p><div className="task-footer"><span className="avatar owner">J</span><span>项目负责人</span></div></article></section><section><div className="kanban-heading"><span>已阻塞</span><Badge tone="danger">1</Badge></div><article className="task-card"><div className="task-top"><span className="mono">T-P1B-RUN</span><Badge tone="danger">门禁</Badge></div><h3>启动 Stage B 筛选实验</h3><p>依赖 Stage A 全部产物和独立审计通过，当前不可执行。</p><div className="task-footer"><span className="avatar muted-avatar">—</span><span>未分配</span></div></article></section></div>; }

function Compute() { return <div className="detail-grid"><section className="panel"><div className="panel-header"><div><span className="eyebrow">LOCAL</span><h2>本地计算环境</h2></div><Badge tone="neutral">空闲</Badge></div><div className="resource-block"><div className="resource-icon"><Code2 /></div><div><strong>Windows · CPU 环境</strong><p>PyTorch 2.5.1+cpu · Python 3.11.15</p></div></div><div className="resource-stats"><div><span>运行中</span><strong>0</strong></div><div><span>成功校准</span><strong>12</strong></div><div><span>资源阻塞</span><strong>1</strong></div></div></section><section className="panel"><div className="panel-header"><div><span className="eyebrow">CLOUD PILOT</span><h2>建议的云端试跑</h2></div><Badge tone="success">预算内</Badge></div><div className="resource-block"><div className="resource-icon cloud"><Cloud /></div><div><strong>RTX 4090 · 24GB</strong><p>仅补跑 A05、A10、A15，不启动 Stage B</p></div></div><div className="budget-row"><div><span>预算上限</span><strong>¥50</strong></div><div><span>建议时限</span><strong>2–4 小时</strong></div><div><span>预计单价</span><strong>约 ¥1.88/小时</strong></div></div><button className="primary-button wide" onClick={() => toast.success("算力审批草案已创建", { description: "预算上限 ¥50；仅限三个 TimesNet 校准任务。" })}>创建算力审批单</button></section></div>; }

const datasets = [
  ["BOND10Y", "Bond / Yield", "2016–2025", "已验证", "6fa4…01c"],
  ["BONDETF", "ETF", "2016–2025", "已验证", "5d2d…9ed"],
  ["CITICSEC", "Equity", "2016–2025", "已验证", "a73c…bb2"],
  ["BTCUSD", "Crypto", "2016–2025", "历史证据", "3ef2…719"],
  ["ETHUSD", "Crypto", "2016–2025", "历史证据", "818a…c40"],
];
const models = [
  ["DeReFusion", "Hybrid residual", "28,436", "可运行", "本项目实现"],
  ["DLinear", "Linear", "4,664", "可运行", "RevIN 包装"],
  ["PatchTST", "Transformer", "284,832", "可运行", "RevIN 包装"],
  ["iTransformer", "Transformer", "278,976", "可运行", "RevIN 包装"],
  ["TimesNet", "Temporal 2D", "—", "资源阻塞", "待 GPU 校准"],
];
const literature = [
  ["RevIN", "Kim et al.", "ICLR 2022", "已复核", "归一化基线"],
  ["DLinear", "Zeng et al.", "AAAI 2023", "已复核", "线性基座"],
  ["PatchTST", "Nie et al.", "ICLR 2023", "已复核", "现代基线"],
  ["TimesNet", "Wu et al.", "ICLR 2023", "已复核", "时序二维建模"],
  ["iTransformer", "Liu et al.", "ICLR 2024", "已复核", "变量注意力"],
];

function RegistryPage({ kind }: { kind: "datasets" | "models" | "literature" }) {
  const configs = {
    datasets: { eyebrow: "DATA GOVERNANCE", title: "数据集注册表", copy: "追踪来源、版本、时间范围、哈希与泄漏审计。", headers: ["名称", "资产类型", "时间范围", "审计状态", "SHA-256"], rows: datasets },
    models: { eyebrow: "MODEL GOVERNANCE", title: "模型注册表", copy: "区分原始方法、本项目实现、参数规模与运行资格。", headers: ["模型", "方法类别", "参数量", "状态", "实现"], rows: models },
    literature: { eyebrow: "KNOWLEDGE BASE", title: "文献知识库", copy: "原文结论、项目解释与复现状态分别记录。", headers: ["论文/方法", "作者", "来源", "复核状态", "项目关系"], rows: literature },
  }[kind];
  return <section className="panel full-panel"><div className="panel-header"><div><span className="eyebrow">{configs.eyebrow}</span><h2>{configs.title}</h2><p className="section-copy">{configs.copy}</p></div><button className="primary-button" onClick={() => toast.info("导入功能将在接入持久化存储后开放")}><Upload size={14} /> 导入</button></div><div className="table-shell"><Table><TableHeader><TableRow>{configs.headers.map((head) => <TableHead key={head}>{head}</TableHead>)}</TableRow></TableHeader><TableBody>{configs.rows.map((row) => <TableRow key={row[0]}>{row.map((cell, index) => <TableCell key={cell} className={index === 0 ? "run-id" : ""}>{index === 3 ? <Badge tone={cell.includes("阻塞") ? "danger" : "success"}>{cell}</Badge> : cell}</TableCell>)}</TableRow>)}</TableBody></Table></div></section>;
}

function Repository() {
  const commits = [["523596c", "添加 Research OS 本地控制台", "本地", "待推送"], ["18b738a", "记录 BOND10Y 控制实验与资源保护", "本地", "待推送"], ["4492900", "新增受控 Phase 1 校准运行器", "本地", "待推送"], ["8412c6b", "记录第二项 Phase 1 校准", "远程", "已同步"]];
  return <div className="detail-grid repo-layout"><section className="panel"><div className="panel-header"><div><span className="eyebrow">REPOSITORY</span><h2>代码仓库状态</h2></div><Badge tone="warning">领先 3</Badge></div><div className="repo-summary"><div><span>分支</span><strong className="mono">main</strong></div><div><span>本地 HEAD</span><strong className="mono">523596c</strong></div><div><span>远程 HEAD</span><strong className="mono">8412c6b</strong></div><div><span>工作区</span><strong>8 个未跟踪结果目录</strong></div></div><div className="sync-warning"><AlertTriangle size={17} /><span>上次远程连接被重置。同步前必须先获取远程更新，禁止强制推送。</span></div></section><section className="panel"><div className="panel-header"><div><span className="eyebrow">COMMITS</span><h2>最近提交</h2></div></div><div className="commit-list">{commits.map((commit) => <div className="commit-row" key={commit[0]}><code>{commit[0]}</code><div><strong>{commit[1]}</strong><small>{commit[2]}</small></div><Badge tone={commit[3] === "已同步" ? "success" : "warning"}>{commit[3]}</Badge></div>)}</div></section></div>;
}

function Reports({ onDownload }: { onDownload: (name: string, body: string) => void }) {
  const reportItems = [{ title: "项目交接报告", desc: "阶段、证据、运行、阻塞与下一步。", ready: true }, { title: "Stage A 校准总结", desc: "12 项通过、1 项资源阻塞、2 项暂停。", ready: true }, { title: "实验审计包", desc: "协议、回执、哈希和环境记录。", ready: false }, { title: "论文实验章节", desc: "确认性证据尚不足，当前禁止生成正式结论。", ready: false }];
  return <section className="panel full-panel"><div className="panel-header"><div><span className="eyebrow">REPORTING</span><h2>报告中心</h2><p className="section-copy">只允许导出能够追溯到证据的内容。</p></div></div><div className="report-grid">{reportItems.map((item) => <article className="report-card" key={item.title}><div className="report-icon"><FileCheck2 /></div><div><h3>{item.title}</h3><p>{item.desc}</p></div><Badge tone={item.ready ? "success" : "neutral"}>{item.ready ? "可生成" : "受门禁限制"}</Badge><button disabled={!item.ready} onClick={() => onDownload(`${item.title}.md`, `# ${item.title}\n\n生成时间：2026-09-28\n\n${item.desc}\n`)}><Download size={14} /> {item.ready ? "生成并下载" : "暂不可用"}</button></article>)}</div></section>;
}

function Audit() {
  const events = [["2026-09-28 13:25", "网站", "创建 Research OS 原型", "523596c", "通过"], ["2026-09-27 16:10", "运行器", "完成 A14 校准与产物打包", "18b738a", "通过"], ["2026-09-27 16:05", "门禁", "TimesNet 模型族自动暂停", "A05", "生效"], ["2026-09-27 15:58", "资源", "A05 超时并记录资源阻塞", "A05", "告警"], ["2026-09-19", "协议", "Phase 1 v1.1 冻结", "6bbb045", "签署"]];
  return <section className="panel full-panel"><div className="panel-header"><div><span className="eyebrow">AUDIT TRAIL</span><h2>不可变审计日志</h2><p className="section-copy">协议、运行、证据与操作事件按时间记录。</p></div><button className="outline-button" onClick={() => toast.success("审计日志完整性检查通过")}><ShieldCheck size={14} /> 验证完整性</button></div><div className="audit-list">{events.map((event) => <div className="audit-row" key={event.join("-")}><span className="audit-time mono">{event[0]}</span><Badge tone="info">{event[1]}</Badge><div><strong>{event[2]}</strong><small className="mono">引用：{event[3]}</small></div><Badge tone={event[4] === "告警" ? "danger" : "success"}>{event[4]}</Badge></div>)}</div></section>;
}

export default function Home() {
  const [view, setView] = useState<View>("overview"); const [sidebarOpen, setSidebarOpen] = useState(false); const [dark, setDark] = useState(false); const [query, setQuery] = useState("");
  const title = allNavigation.find((item) => item.id === view)?.label ?? "项目总览";
  const searchMatches = query.trim() ? [...runs.map((run) => ({ title: `${run.id} · ${run.model}`, note: `${run.dataset} · ${run.status}`, view: "experiments" as View })), ...literature.map((paper) => ({ title: paper[0], note: `${paper[1]} · ${paper[2]}`, view: "literature" as View }))].filter((item) => `${item.title} ${item.note}`.toLowerCase().includes(query.toLowerCase())).slice(0, 6) : [];
  const downloadText = (name: string, body: string) => { const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([body], { type: "text/markdown;charset=utf-8" })); link.download = name; link.click(); URL.revokeObjectURL(link.href); toast.success(`${name} 已生成`); };
  useEffect(() => {
    type WebMCPContext = { registerTool: (tool: object, options?: { signal?: AbortSignal }) => void | Promise<void> };
    const context = (document as Document & { modelContext?: WebMCPContext }).modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    void Promise.resolve(context.registerTool({
      name: "navigate_research_view",
      title: "切换研究视图",
      description: "在项目总览、实验、协议、证据、任务和算力视图之间切换。",
      inputSchema: { type: "object", properties: { view: { type: "string", enum: ["overview", "experiments", "protocols", "evidence", "tasks", "compute", "datasets", "models", "literature", "repository", "reports", "audit"] } }, required: ["view"], additionalProperties: false },
      annotations: { readOnlyHint: true, untrustedContentHint: false },
      execute(input: unknown) {
        const next = (input as { view?: View })?.view;
        if (!next || !nav.some((item) => item.id === next)) throw new Error("未知研究视图");
        setView(next); return { activeView: next };
      },
    }, { signal: lifecycle.signal })).catch(() => undefined);
    return () => lifecycle.abort();
  }, []);
  return <div className={dark ? "app dark" : "app"}>
    <header className="topbar"><button className="mobile-menu" aria-label="打开菜单" onClick={() => setSidebarOpen(true)}><Menu /></button><div className="brand"><div className="brand-mark"><Microscope size={19} /></div><strong>Research OS</strong><span className="version">INSTITUTIONAL</span></div><button className="project-picker" onClick={() => toast.info("当前工作区：DeReFusion")}><span className="project-code">DR</span><span><strong>DeReFusion</strong><small>研究工作区</small></span><ChevronDown size={16} /></button><div className="search-wrap"><label className="global-search"><Search size={17} /><input aria-label="全局搜索" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索实验、主张、任务或文献…" /><kbd>⌘ K</kbd></label>{query && <div className="search-results">{searchMatches.length ? searchMatches.map((item) => <button key={`${item.title}-${item.note}`} onClick={() => { setView(item.view); setQuery(""); }}><strong>{item.title}</strong><span>{item.note}</span></button>) : <div className="search-empty">没有匹配结果</div>}</div>}</div><div className="top-actions"><button aria-label="切换主题" onClick={() => setDark(!dark)}>{dark ? <Sun /> : <Moon />}</button><button aria-label="帮助" onClick={() => toast.info("Research OS", { description: "所有正式结论必须能追溯到协议、实验和原始产物。" })}><HelpCircle /></button><div className="user-avatar">J</div></div></header>
    <aside className={sidebarOpen ? "sidebar open" : "sidebar"}><button className="close-sidebar" aria-label="关闭菜单" onClick={() => setSidebarOpen(false)}><X /></button><nav aria-label="项目导航"><span className="nav-label">研究工作</span>{nav.map((item) => { const Icon = item.icon; return <button key={item.id} className={view === item.id ? "nav-item active" : "nav-item"} onClick={() => { setView(item.id as View); setSidebarOpen(false); }}><Icon size={17} /><span>{item.label}</span>{item.count && <em>{item.count}</em>}</button>; })}<span className="nav-label secondary-label">研究资产</span>{secondary.map((item) => { const Icon = item.icon; return <button key={item.label} className={view === item.id ? "nav-item active" : "nav-item muted-nav"} onClick={() => { setView(item.id as View); setSidebarOpen(false); }}><Icon size={17} /><span>{item.label}</span></button>; })}</nav><div className="sidebar-footer"><div className="sync-row"><span className="sync-dot" /><div><strong>本地领先 3 个提交</strong><small>远程同步待处理</small></div></div><button onClick={() => toast.info("设置模块将在下一版本接入")}><Settings size={16} /> 工作区设置</button></div></aside>
    <main className="content"><div className="page-heading"><div><div className="breadcrumbs">研究工作区 <span>/</span> DeReFusion <span>/</span> {title}</div><h1>{title}</h1><p>{view === "overview" ? "证据优先的研究进度、风险和下一步决策。" : "DeReFusion · Phase 1 研究记录"}</p></div><div className="heading-actions"><Badge tone="info">截至 2026-09-28</Badge><button className="outline-button" onClick={() => downloadText("DeReFusion-项目交接报告.md", "# DeReFusion 项目交接报告\n\nStage A：12/15 校准通过；TimesNet 资源阻塞。\n\nStage B：门禁锁定。\n")}><ClipboardCheck size={15} /> 生成交接报告</button></div></div>{view === "overview" && <Overview />}{view === "experiments" && <Experiments />}{view === "protocols" && <Protocols />}{view === "evidence" && <Evidence />}{view === "tasks" && <Tasks />}{view === "compute" && <Compute />}{view === "datasets" && <RegistryPage kind="datasets" />}{view === "models" && <RegistryPage kind="models" />}{view === "literature" && <RegistryPage kind="literature" />}{view === "repository" && <Repository />}{view === "reports" && <Reports onDownload={downloadText} />}{view === "audit" && <Audit />}</main><Toaster position="bottom-right" richColors />
  </div>;
}
