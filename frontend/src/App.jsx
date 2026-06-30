import { useMemo, useState } from "react";
import {
  BrainCircuit,
  BriefcaseBusiness,
  CheckCircle2,
  Database,
  ExternalLink,
  Eye,
  FileJson,
  FolderKanban,
  Gauge,
  Github,
  GraduationCap,
  Link as LinkIcon,
  ListChecks,
  Loader2,
  Mail,
  MapPin,
  Phone,
  Play,
  Repeat2,
  Sparkles,
  Trophy,
  UploadCloud,
  X,
  XCircle
} from "lucide-react";

const MAX_FILES = 5;
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = [".csv", ".json", ".txt", ".md", ".pdf", ".docx"];
const CUSTOM_FIELD_OPTIONS = [
  { group: "Identity", label: "Full Name", path: "full_name", type: "string" },
  { group: "Identity", label: "Headline", path: "headline", type: "string" },
  { group: "Identity", label: "Years Experience", path: "years_experience", type: "number" },
  { group: "Contact", label: "Emails", path: "emails", type: "string[]" },
  { group: "Contact", label: "Phones", path: "phones", type: "string[]" },
  { group: "Location", label: "Location", path: "location", type: "object" },
  { group: "Location", label: "City", path: "location.city", type: "string" },
  { group: "Location", label: "Region", path: "location.region", type: "string" },
  { group: "Location", label: "Country", path: "location.country", type: "string" },
  { group: "Links", label: "GitHub URL", path: "links.github", type: "string" },
  { group: "Links", label: "LinkedIn URL", path: "links.linkedin", type: "string" },
  { group: "Links", label: "Portfolio URL", path: "links.portfolio", type: "string" },
  { group: "Links", label: "Other Links", path: "links.other", type: "string[]" },
  { group: "Recruiting", label: "Education", path: "education", type: "object[]" },
  { group: "Recruiting", label: "Experience", path: "experience", type: "object[]" },
  { group: "Recruiting", label: "Projects", path: "projects", type: "object[]" },
  { group: "Recruiting", label: "Skills", path: "skills", type: "object[]" },
  { group: "Recruiting", label: "Achievements", path: "achievements", type: "object[]" },
  { group: "Recruiting", label: "Certifications", path: "certifications" },
  { group: "Recruiting", label: "Publications", path: "publications" },
  { group: "Recruiting", label: "Online Coding Profile", path: "online_coding_profile", type: "object" },
  { group: "Recruiting", label: "GitHub Repositories", path: "github_repositories", type: "object[]" },
  { group: "Recruiting", label: "Languages", path: "languages" },
  { group: "Recruiting", label: "Extracurriculars", path: "extracurriculars" },
  { group: "Recruiting", label: "Profile Summary", path: "profile_summary", type: "string" },
  { group: "Recruiting", label: "Other Sections", path: "other_sections", type: "object[]" },
  { group: "Recruiting", label: "Others", path: "others", type: "object[]" },
  { group: "Diagnostics", label: "Overall Confidence", path: "overall_confidence", type: "number" },
  { group: "Diagnostics", label: "Provenance", path: "provenance", type: "object[]" },
  { group: "Diagnostics", label: "Resume Sections", path: "resume_sections", type: "object" },
  { group: "Diagnostics", label: "Semantic Mappings", path: "semantic_mappings", type: "object[]" },
  { group: "Diagnostics", label: "Extraction Errors", path: "extraction_errors", type: "string[]" },
  { group: "Metadata", label: "Candidate ID", path: "candidate_id", type: "string" }
];
const DEFAULT_SELECTED_FIELDS = Object.fromEntries(CUSTOM_FIELD_OPTIONS.map((field) => [field.path, true]));
const DEFAULT_FIELD_RENAMES = Object.fromEntries(CUSTOM_FIELD_OPTIONS.map((field) => [field.path, field.path]));
const DEFAULT_KEEP_EMPTY_FIELDS = Object.fromEntries(CUSTOM_FIELD_OPTIONS.map((field) => [field.path, false]));

function JsonPanel({ title, data }) {
  return (
    <section className="rounded-lg border border-line bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <FileJson size={17} />
          {title}
        </div>
      </div>
      <pre className="max-h-[560px] overflow-auto p-4 text-xs leading-5 text-slate-800">
        {data ? JSON.stringify(data, null, 2) : "{\n  \"waiting\": true\n}"}
      </pre>
    </section>
  );
}

const BULLET_MARKER_RE = /(?:\u00c2\u2022|\u0100\u2022|\u0095|\u2022|Â•|Ā•)/g;
const BULLET_SPLIT_RE = /\s*(?:\u2022|\u0095)\s+/g;

function normalizeDisplayText(value) {
  return String(value || "")
    .replace(BULLET_MARKER_RE, " • ")
    .replace(/\s+/g, " ")
    .trim();
}

function cleanInlineText(value) {
  return normalizeDisplayText(value).replace(BULLET_SPLIT_RE, " ").replace(/\s+/g, " ").trim();
}

function splitBulletText(value) {
  const normalized = normalizeDisplayText(value);
  if (!normalized) return { intro: "", items: [] };
  if (!normalized.includes("•") && !/\s\?\s+[A-Z0-9]/.test(normalized)) {
    return { intro: cleanInlineText(normalized), items: [] };
  }
  const parts = normalized
    .replace(/\s\?\s+(?=[A-Z0-9])/g, " • ")
    .split(BULLET_SPLIT_RE)
    .map(cleanInlineText)
    .filter(Boolean);
  const startsWithBullet = normalized.trim().startsWith("•") || normalized.trim().startsWith("\u0095");
  return {
    intro: startsWithBullet ? "" : parts.shift() || "",
    items: parts
  };
}

function TextWithBullets({ text, compact = false }) {
  const { intro, items } = splitBulletText(text);
  if (!intro && !items.length) return null;
  return (
    <div className={compact ? "text-sm leading-5 text-slate-600" : "text-sm leading-6 text-slate-700"}>
      {intro && <p>{intro}</p>}
      {items.length > 0 && (
        <ul className={`${intro ? "mt-2" : ""} list-disc space-y-1 pl-5`}>
          {items.map((item, index) => (
            <li key={`${item}-${index}`}>{item}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function InfoItem({ icon: Icon, label, value }) {
  if (!value) return null;
  return (
    <div className="flex min-w-0 items-start gap-2 rounded-md border border-line bg-slate-50 px-3 py-2">
      <Icon size={16} className="mt-0.5 shrink-0 text-slate-500" />
      <div className="min-w-0">
        <div className="text-[11px] font-semibold uppercase tracking-normal text-slate-500">{label}</div>
        <div className="break-words text-sm text-slate-800">{cleanInlineText(value)}</div>
      </div>
    </div>
  );
}

function SkillChips({ skills = [] }) {
  if (!skills.length) return null;
  return (
    <div className="flex flex-wrap gap-2">
      {skills.slice(0, 42).map((skill) => (
        <span key={skill.name} className="rounded-full border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700">
          {skill.name}
        </span>
      ))}
    </div>
  );
}

function TextChips({ items = [] }) {
  const cleaned = [...new Set(items.map(cleanInlineText).filter(Boolean))];
  if (!cleaned.length) return null;
  return (
    <div className="flex flex-wrap gap-2">
      {cleaned.map((item) => (
        <span key={item} className="rounded-full border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700">
          {item}
        </span>
      ))}
    </div>
  );
}

function buildCustomConfig(selectedFields = {}, fieldRenames = {}, keepEmptyFields = {}) {
  return {
    fields: CUSTOM_FIELD_OPTIONS.filter((field) => selectedFields[field.path]).map((field) => {
      const targetPath = (fieldRenames[field.path] || field.path).trim() || field.path;
      const outputField = {
        path: targetPath,
        from: field.path,
        on_missing: keepEmptyFields[field.path] ? "null" : "omit"
      };
      if (field.type) outputField.type = field.type;
      if (field.required) outputField.required = true;
      return outputField;
    }),
    on_missing: "omit"
  };
}

function groupedCustomFields() {
  return CUSTOM_FIELD_OPTIONS.reduce((groups, field) => {
    if (!groups[field.group]) groups[field.group] = [];
    groups[field.group].push(field);
    return groups;
  }, {});
}

function MergeTrust({ profile }) {
  if (!profile?.provenance?.length) return null;
  const rows = [...profile.provenance]
    .sort((left, right) => (right.confidence || 0) - (left.confidence || 0))
    .slice(0, 18);
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
        <ListChecks size={16} />
        Merge & Trust
      </div>
      <div className="overflow-hidden rounded-md border border-line">
        <div className="grid grid-cols-[1.1fr_1.4fr_1.2fr_80px] border-b border-line bg-orange-50 px-3 py-2 text-xs font-semibold uppercase tracking-normal text-slate-600">
          <div>Field</div>
          <div>Source</div>
          <div>Method</div>
          <div className="text-right">Conf.</div>
        </div>
        <div className="max-h-72 overflow-auto divide-y divide-line">
          {rows.map((row, index) => (
            <div key={`${row.field}-${row.source}-${index}`} className="grid grid-cols-[1.1fr_1.4fr_1.2fr_80px] gap-2 px-3 py-2 text-xs text-slate-700">
              <div className="font-medium text-slate-900">{cleanInlineText(row.field)}</div>
              <div className="break-words">{cleanInlineText(row.source)}</div>
              <div className="break-words">{cleanInlineText(row.method)}</div>
              <div className="text-right font-semibold">{Number(row.confidence || 0).toFixed(2)}</div>
              {row.evidence && <div className="col-span-4 text-slate-500">Evidence: {cleanInlineText(row.evidence)}</div>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SemanticMappings({ profile }) {
  const rows = (profile?.semantic_mappings || []).filter((row) => row.method?.startsWith("gemini"));
  if (!rows.length) return null;
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
        <BrainCircuit size={16} />
        Gemini Semantic Decisions
      </div>
      <div className="overflow-hidden rounded-md border border-line">
        <div className="grid grid-cols-[1.2fr_0.8fr_1fr_70px_70px] border-b border-line bg-orange-50 px-3 py-2 text-xs font-semibold uppercase tracking-normal text-slate-600">
          <div>Input</div>
          <div>Kind</div>
          <div>Mapped To</div>
          <div className="text-right">Conf.</div>
          <div className="text-right">Used</div>
        </div>
        <div className="max-h-72 overflow-auto divide-y divide-line">
          {rows.slice(0, 30).map((row, index) => (
            <div key={`${row.original}-${index}`} className="grid grid-cols-[1.2fr_0.8fr_1fr_70px_70px] gap-2 px-3 py-2 text-xs text-slate-700">
              <div className="break-words font-medium text-slate-900">{cleanInlineText(row.original)}</div>
              <div>{cleanInlineText(row.kind)}</div>
              <div className="break-words">{cleanInlineText(row.mapped_to || row.canonical_section || "-")}</div>
              <div className="text-right font-semibold">{Number(row.confidence || 0).toFixed(2)}</div>
              <div className="text-right">{row.applied ? "yes" : "no"}</div>
              {row.reason && <div className="col-span-5 text-slate-500">{cleanInlineText(row.reason)}</div>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function scoreClass(score) {
  if (score >= 8) return "border-emerald-200 bg-emerald-50 text-emerald-700";
  if (score >= 6) return "border-amber-200 bg-amber-50 text-amber-700";
  return "border-red-200 bg-red-50 text-red-700";
}

function JsonBlock({ data }) {
  return (
    <pre className="max-h-72 overflow-auto rounded-md border border-line bg-slate-950 p-3 text-xs leading-5 text-slate-100">
      {JSON.stringify(data || {}, null, 2)}
    </pre>
  );
}

function PromptBlock({ title, text }) {
  if (!text) return null;
  return (
    <div>
      <div className="mb-1 text-xs font-semibold uppercase tracking-normal text-slate-500">{title}</div>
      <pre className="max-h-56 overflow-auto whitespace-pre-wrap rounded-md border border-line bg-white p-3 text-xs leading-5 text-slate-700">
        {text}
      </pre>
    </div>
  );
}

function AgentTraceModal({ trace, onClose }) {
  if (!trace) return null;
  const iterations = trace.iterations || [];
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4">
      <div className="flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-lg border border-line bg-white shadow-xl">
        <div className="flex items-start justify-between border-b border-line px-4 py-3">
          <div>
            <div className="text-base font-semibold">{cleanInlineText(trace.task_name)}</div>
            <div className="mt-1 text-xs text-slate-500">
              {cleanInlineText(trace.mode)} | loops {trace.loops || iterations.length || 0} | score {Number(trace.final_score || 0).toFixed(2)} | {trace.accepted ? "accepted" : "discarded"}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            title="Close trace"
            className="rounded-md border border-line p-2 text-slate-600 hover:border-ink hover:text-ink"
          >
            <X size={17} />
          </button>
        </div>

        <div className="space-y-4 overflow-auto p-4">
          <div className="grid gap-3 md:grid-cols-4">
            <div className={`rounded-md border px-3 py-2 ${scoreClass(Number(trace.final_score || 0))}`}>
              <div className="text-[11px] font-semibold uppercase tracking-normal">Final Score</div>
              <div className="text-lg font-semibold">{Number(trace.final_score || 0).toFixed(2)}/10</div>
            </div>
            <div className="rounded-md border border-line bg-slate-50 px-3 py-2">
              <div className="text-[11px] font-semibold uppercase tracking-normal text-slate-500">Mode</div>
              <div className="text-sm font-medium text-slate-800">{cleanInlineText(trace.mode)}</div>
            </div>
            <div className="rounded-md border border-line bg-slate-50 px-3 py-2">
              <div className="text-[11px] font-semibold uppercase tracking-normal text-slate-500">Status</div>
              <div className="text-sm font-medium text-slate-800">{trace.passed ? "passed" : "not passed"}</div>
            </div>
            <div className="rounded-md border border-line bg-slate-50 px-3 py-2">
              <div className="text-[11px] font-semibold uppercase tracking-normal text-slate-500">Stop</div>
              <div className="text-sm font-medium text-slate-800">{cleanInlineText(trace.stopping_reason)}</div>
            </div>
          </div>

          <div>
            <div className="mb-1 text-xs font-semibold uppercase tracking-normal text-slate-500">Purpose</div>
            <div className="rounded-md border border-line bg-slate-50 p-3 text-sm text-slate-700">{cleanInlineText(trace.purpose)}</div>
          </div>

          <div className="grid gap-3 xl:grid-cols-2">
            <PromptBlock title="System Prompt" text={trace.system_prompt} />
            <PromptBlock title="Evaluator Prompt" text={trace.evaluator_prompt} />
          </div>

          {iterations.length > 0 && (
            <div>
              <div className="mb-2 text-sm font-semibold">Intermediate Steps</div>
              <div className="space-y-3">
                {iterations.map((step) => (
                  <div key={step.loop} className="rounded-md border border-line bg-slate-50 p-3">
                    <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                      <span className="rounded-full border border-slate-300 bg-white px-2 py-1 font-semibold">Loop {step.loop}</span>
                      <span className={`rounded-full border px-2 py-1 font-semibold ${scoreClass(Number(step.score || 0))}`}>Score {Number(step.score || 0).toFixed(2)}</span>
                      <span className="rounded-full border border-slate-300 bg-white px-2 py-1">{step.passed ? "passed" : "not passed"}</span>
                      <span className="rounded-full border border-slate-300 bg-white px-2 py-1">{cleanInlineText(step.action)}</span>
                    </div>
                    {step.rationale_summary && (
                      <div className="mb-2 text-sm text-slate-700">
                        <span className="font-semibold">Reasoning summary: </span>
                        {cleanInlineText(step.rationale_summary)}
                      </div>
                    )}
                    {step.observation && (
                      <div className="mb-2 text-sm text-slate-700">
                        <span className="font-semibold">Observation: </span>
                        {typeof step.observation === "string" ? cleanInlineText(step.observation) : JSON.stringify(step.observation)}
                      </div>
                    )}
                    <div className="grid gap-3 xl:grid-cols-2">
                      <div>
                        <div className="mb-1 text-xs font-semibold uppercase tracking-normal text-slate-500">Candidate Output</div>
                        <JsonBlock data={step.candidate_output || step.observation} />
                      </div>
                      <div>
                        <div className="mb-1 text-xs font-semibold uppercase tracking-normal text-slate-500">Evaluator Output</div>
                        <JsonBlock data={step.evaluation} />
                      </div>
                    </div>
                    {step.request_events?.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {step.request_events.map((event, eventIndex) => (
                          <span key={`${event.task}-${eventIndex}`} className="rounded-full border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-700">
                            {cleanInlineText(event.task)} | key {event.key_index ?? "-"} | {event.status || "error"} | {event.seconds ?? "-"}s
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {trace.final_output && (
            <div>
              <div className="mb-1 text-xs font-semibold uppercase tracking-normal text-slate-500">Accepted Final Output</div>
              <JsonBlock data={trace.final_output} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function AgentOps({ diagnostics }) {
  const [selectedTrace, setSelectedTrace] = useState(null);
  const llmops = diagnostics;
  if (!llmops) {
    return (
      <section className="rounded-lg border border-line bg-white">
        <div className="border-b border-line px-4 py-3">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Gauge size={16} />
            Agent Traces
          </div>
        </div>
        <div className="p-4 text-sm text-slate-600">No agent diagnostics returned for this transform.</div>
      </section>
    );
  }
  const iterations = llmops.iterations || [];
  const events = llmops.request_events || [];
  const taskTraces = llmops.task_traces || [];
  const finalScore = Number(llmops.final_score || 0);
  return (
    <section className="rounded-lg border border-line bg-white">
      <div className="border-b border-line px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <Gauge size={16} />
          Agent Traces
        </div>
      </div>
      <div className="space-y-3 p-4">
        <div className="grid gap-2 md:grid-cols-4">
          <div className={`rounded-md border px-3 py-2 ${scoreClass(finalScore)}`}>
            <div className="text-[11px] font-semibold uppercase tracking-normal">Score</div>
            <div className="text-lg font-semibold">{finalScore.toFixed(2)}/10</div>
          </div>
          <div className="rounded-md border border-line bg-white px-3 py-2">
            <div className="text-[11px] font-semibold uppercase tracking-normal text-slate-500">Mode</div>
            <div className="truncate text-sm font-medium text-slate-800">{cleanInlineText(llmops.mode)}</div>
          </div>
          <div className="rounded-md border border-line bg-white px-3 py-2">
            <div className="text-[11px] font-semibold uppercase tracking-normal text-slate-500">Stop</div>
            <div className="truncate text-sm font-medium text-slate-800">{cleanInlineText(llmops.stopping_reason)}</div>
          </div>
          <div className="rounded-md border border-line bg-white px-3 py-2">
            <div className="text-[11px] font-semibold uppercase tracking-normal text-slate-500">Memory</div>
            <div className="text-sm font-medium text-slate-800">{llmops.memory_examples_used || 0} examples</div>
          </div>
        </div>

        {llmops.tasks?.length > 0 && (
          <div>
            <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-normal text-slate-500">
              <Repeat2 size={13} />
              Decomposed Tasks
            </div>
            <div className="grid gap-2 md:grid-cols-2">
              {llmops.tasks.slice(0, 6).map((task, index) => (
                <div key={`${task.name}-${index}`} className="rounded-md border border-line bg-white px-3 py-2 text-xs">
                  <div className="font-semibold text-slate-800">{cleanInlineText(task.name)}</div>
                  <div className="mt-1 text-slate-600">{cleanInlineText(task.purpose)}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {taskTraces.length > 0 && (
          <div>
            <div className="mb-1 text-xs font-semibold uppercase tracking-normal text-slate-500">Task Execution Traces</div>
            <div className="grid gap-2 xl:grid-cols-2">
              {taskTraces.map((trace, index) => (
                <div key={`${trace.task_name}-${index}`} className="rounded-md border border-line bg-white px-3 py-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold text-slate-900">{cleanInlineText(trace.task_name)}</div>
                      <div className="mt-1 text-xs text-slate-500">
                        {cleanInlineText(trace.mode)} | loops {trace.loops || 0} | {trace.accepted ? "accepted" : "discarded"}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      <span className={`rounded-full border px-2 py-1 text-xs font-semibold ${scoreClass(Number(trace.final_score || 0))}`}>
                        {Number(trace.final_score || 0).toFixed(2)}
                      </span>
                      <button
                        type="button"
                        onClick={() => setSelectedTrace(trace)}
                        title="View trace"
                        className="rounded-md border border-line p-1.5 text-slate-600 hover:border-ink hover:text-ink"
                      >
                        <Eye size={15} />
                      </button>
                    </div>
                  </div>
                  <div className="mt-2 text-xs leading-5 text-slate-600">{cleanInlineText(trace.purpose)}</div>
                  <div className="mt-2 text-xs text-slate-500">Stop: {cleanInlineText(trace.stopping_reason)}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {iterations.length > 0 && (
          <div className="overflow-hidden rounded-md border border-line bg-white">
            <div className="grid grid-cols-[60px_90px_1fr] border-b border-line bg-slate-50 px-3 py-2 text-xs font-semibold uppercase tracking-normal text-slate-500">
              <div>Loop</div>
              <div>Score</div>
              <div>Evaluator Notes</div>
            </div>
            <div className="divide-y divide-line">
              {iterations.map((item) => (
                <div key={item.loop} className="grid grid-cols-[60px_90px_1fr] gap-2 px-3 py-2 text-xs text-slate-700">
                  <div className="font-semibold">{item.loop}</div>
                  <div className="font-semibold">{Number(item.score || 0).toFixed(2)}</div>
                  <div>
                    <div>{cleanInlineText(item.evaluation?.verdict || "No verdict")}</div>
                    {item.applied_changes?.length > 0 && (
                      <div className="mt-1 text-slate-500">
                        Applied: {item.applied_changes.map((change) => cleanInlineText(change.value || change.field)).join(", ")}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {events.length > 0 && (
          <div>
            <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-normal text-slate-500">
              <Database size={13} />
              Gemini Calls
            </div>
            <div className="flex flex-wrap gap-2">
              {events.slice(0, 16).map((event, index) => (
                <span key={`${event.task}-${index}`} className="rounded-full border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-700">
                  {cleanInlineText(event.task)} | key {event.key_index ?? "-"} | {event.status || "error"} | {event.seconds ?? "-"}s
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
      <AgentTraceModal trace={selectedTrace} onClose={() => setSelectedTrace(null)} />
    </section>
  );
}

function CleanProfile({ profile }) {
  if (!profile) {
    return (
      <section className="rounded-lg border border-line bg-white p-4">
        <div className="text-sm text-slate-600">Run a transform to see the extracted profile.</div>
      </section>
    );
  }

  const primaryEmail = profile.emails?.[0];
  const primaryPhone = profile.phones?.[0];
  const links = profile.links || {};
  const visibleSections = Object.entries(profile.resume_sections || {}).filter(
    ([name]) => !["Education", "Experience", "Projects", "Skills", "Skills Summary"].includes(name)
  );

  return (
    <section className="rounded-lg border border-line bg-white">
      <div className="border-b border-line px-4 py-3">
        <div className="text-lg font-semibold">{profile.full_name || "Unknown Candidate"}</div>
        {profile.headline && <div className="mt-1 text-sm text-slate-600">{profile.headline}</div>}
      </div>

      <div className="space-y-5 p-4">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <InfoItem icon={Mail} label="Email" value={primaryEmail} />
          <InfoItem icon={Phone} label="Phone" value={primaryPhone} />
          <InfoItem icon={Github} label="GitHub" value={links.github} />
          <InfoItem icon={LinkIcon} label="LinkedIn" value={links.linkedin} />
        </div>

        {profile.profile_summary && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
              <Sparkles size={16} />
              Summary
            </div>
            <div className="rounded-md border border-line bg-slate-50 px-3 py-3">
              <TextWithBullets text={profile.profile_summary} />
            </div>
          </div>
        )}

        {profile.education?.length > 0 && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
              <GraduationCap size={16} />
              Education
            </div>
            <div className="divide-y divide-line rounded-md border border-line">
              {profile.education.map((item, index) => (
                <div key={`${item.institution}-${index}`} className="px-3 py-3">
                  <div className="font-medium text-slate-900">{cleanInlineText(item.institution)}</div>
                  <div className="mt-1 text-sm text-slate-600">
                    {[item.degree, item.field, item.end_year && `Ends ${item.end_year}`, item.cgpa && `CGPA ${item.cgpa}`]
                      .filter(Boolean)
                      .map(cleanInlineText)
                      .join(" - ")}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {profile.experience?.length > 0 && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
              <BriefcaseBusiness size={16} />
              Experience
            </div>
            <div className="divide-y divide-line rounded-md border border-line">
              {profile.experience.map((item, index) => (
                <div key={`${item.company}-${item.title}-${index}`} className="px-3 py-3">
                  <div className="flex flex-col gap-1 md:flex-row md:items-start md:justify-between">
                    <div>
                      <div className="font-medium text-slate-900">{cleanInlineText(item.role || item.title)}</div>
                      <div className="text-sm text-slate-700">{cleanInlineText(item.company)}</div>
                    </div>
                    <div className="text-sm text-slate-600 md:text-right">
                      {item.duration && <div>{cleanInlineText(item.duration)}</div>}
                      {item.location && (
                        <div className="mt-1 flex items-center gap-1 md:justify-end">
                          <MapPin size={14} />
                          {cleanInlineText(item.location)}
                        </div>
                      )}
                    </div>
                  </div>
                  {item.summary && <div className="mt-2"><TextWithBullets text={item.summary} compact /></div>}
                </div>
              ))}
            </div>
          </div>
        )}

        {profile.projects?.length > 0 && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
              <FolderKanban size={16} />
              Projects
            </div>
            <div className="divide-y divide-line rounded-md border border-line">
              {profile.projects.map((project, index) => (
                <div key={`${project.title}-${index}`} className="space-y-3 px-3 py-3">
                  <div className="flex flex-col gap-1 md:flex-row md:items-start md:justify-between">
                    <div className="font-medium text-slate-900">{cleanInlineText(project.title)}</div>
                    {project.date && <div className="text-sm text-slate-600">{cleanInlineText(project.date)}</div>}
                  </div>
                  {project.tech_stack?.length > 0 && (
                    <div>
                      <div className="mb-1 text-xs font-semibold uppercase tracking-normal text-slate-500">Tech Stack</div>
                      <TextChips items={project.tech_stack} />
                    </div>
                  )}
                  {project.links?.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {project.links.map((link) => (
                        <a
                          key={link}
                          href={link}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 rounded-md border border-line px-2.5 py-1 text-xs font-medium text-slate-700 hover:border-ink"
                        >
                          <ExternalLink size={13} />
                          {cleanInlineText(link)}
                        </a>
                      ))}
                    </div>
                  )}
                  {project.bullets?.length > 0 && (
                    <ul className="list-disc space-y-1 pl-5 text-sm leading-5 text-slate-600">
                      {project.bullets.map((bullet, bulletIndex) => (
                        <li key={`${project.title}-bullet-${bulletIndex}`}>{cleanInlineText(bullet)}</li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {profile.achievements?.length > 0 && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
              <Trophy size={16} />
              Achievements
            </div>
            <div className="divide-y divide-line rounded-md border border-line">
              {profile.achievements.map((achievement, index) => (
                <div key={`${achievement.title}-${index}`} className="space-y-2 px-3 py-3">
                  <div className="font-medium text-slate-900">{cleanInlineText(achievement.title)}</div>
                  {achievement.summary && <TextWithBullets text={achievement.summary} compact />}
                  {achievement.links?.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {achievement.links.map((link) => (
                        <a
                          key={link}
                          href={link}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 rounded-md border border-line px-2.5 py-1 text-xs font-medium text-slate-700 hover:border-ink"
                        >
                          <ExternalLink size={13} />
                          {cleanInlineText(link)}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {visibleSections.length > 0 && (
          <div>
            <div className="mb-2 text-sm font-semibold">Other Sections</div>
            <div className="grid gap-3 xl:grid-cols-2">
              {visibleSections.map(([name, body]) => (
                <div key={name} className="rounded-md border border-line px-3 py-3">
                  <div className="mb-1 text-sm font-semibold">{name}</div>
                  <TextWithBullets text={body} compact />
                </div>
              ))}
            </div>
          </div>
        )}

        {profile.others?.length > 0 && (
          <div>
            <div className="mb-2 text-sm font-semibold">Others</div>
            <div className="grid gap-3 xl:grid-cols-2">
              {profile.others.map((item, index) => (
                <div key={`${item.title || "other"}-${index}`} className="rounded-md border border-line px-3 py-3">
                  <div className="mb-1 text-sm font-semibold">{cleanInlineText(item.title || "Other")}</div>
                  <TextWithBullets text={typeof item.content === "string" ? item.content : JSON.stringify(item.content)} compact />
                </div>
              ))}
            </div>
          </div>
        )}

        <div>
          <div className="mb-2 text-sm font-semibold">Skills</div>
          <SkillChips skills={profile.skills || []} />
        </div>

        <MergeTrust profile={profile} />
        <SemanticMappings profile={profile} />
      </div>
    </section>
  );
}

function FieldCheckboxGroups({ title, description, fields, checkedMap, disabledMap = {}, onToggle }) {
  return (
    <div>
      <div className="mb-2">
        <div className="text-xs font-semibold uppercase tracking-normal text-slate-500">{title}</div>
        {description && <div className="mt-1 text-xs leading-5 text-slate-500">{description}</div>}
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {fields.map((field) => (
          <label
            key={field.path}
            className={`flex min-w-0 items-start gap-2 rounded-md border border-line px-2.5 py-2 text-sm ${
              disabledMap[field.path] ? "bg-slate-100 text-slate-400" : "bg-slate-50 text-slate-800"
            }`}
          >
            <input
              className="mt-1"
              type="checkbox"
              checked={Boolean(checkedMap[field.path])}
              disabled={Boolean(disabledMap[field.path])}
              onChange={(event) => onToggle(field.path, event.target.checked)}
            />
            <span className="min-w-0">
              <span className="block font-medium">{field.label}</span>
              <code className="block truncate text-xs text-slate-500">{field.path}</code>
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}

function CustomFieldSelector({
  selectedFields,
  keepEmptyFields,
  fieldRenames,
  onToggle,
  onToggleKeepEmpty,
  onRename,
  onSelectAll,
  onClearAll,
  configPreview
}) {
  const groups = groupedCustomFields();
  const selectedCount = CUSTOM_FIELD_OPTIONS.filter((field) => selectedFields[field.path]).length;
  const disabledEmptyFields = Object.fromEntries(CUSTOM_FIELD_OPTIONS.map((field) => [field.path, !selectedFields[field.path]]));
  return (
    <section className="rounded-md border border-line">
      <div className="flex items-center justify-between gap-3 border-b border-line px-3 py-2">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <ListChecks size={16} />
          Custom Output Fields
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-slate-500">{selectedCount}/{CUSTOM_FIELD_OPTIONS.length}</span>
          <button
            type="button"
            onClick={onSelectAll}
            title="Select all fields"
            className="rounded-md border border-line px-2 py-1 text-xs font-medium text-slate-700 hover:border-ink"
          >
            All
          </button>
          <button
            type="button"
            onClick={onClearAll}
            title="Clear all fields"
            className="rounded-md border border-line p-1.5 text-slate-600 hover:border-ink"
          >
            <X size={14} />
          </button>
        </div>
      </div>
      <div className="max-h-[520px] space-y-5 overflow-auto p-3">
        {Object.entries(groups).map(([group, fields]) => (
          <div key={group} className="space-y-4 rounded-md border border-line bg-white p-3">
            <div className="text-sm font-semibold text-slate-800">{group}</div>
            <FieldCheckboxGroups
              title="1. Include In Custom Output"
              description="Select the canonical fields that should be present in the generated custom structure."
              fields={fields}
              checkedMap={selectedFields}
              onToggle={onToggle}
            />
            <FieldCheckboxGroups
              title="2. Keep If Empty Or Null"
              description="Enable this only for selected fields that must still appear as null when no value is extracted."
              fields={fields}
              checkedMap={keepEmptyFields}
              disabledMap={disabledEmptyFields}
              onToggle={onToggleKeepEmpty}
            />
            <div>
              <div className="mb-2">
                <div className="text-xs font-semibold uppercase tracking-normal text-slate-500">3. Rename Output Field</div>
                <div className="mt-1 text-xs leading-5 text-slate-500">Aliases affect only custom output keys; canonical names remain unchanged inside the engine.</div>
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {fields.map((field) => (
                  <label key={field.path} className="block rounded-md border border-line bg-slate-50 px-2.5 py-2">
                    <span className="mb-1 block truncate text-xs font-medium text-slate-600">{field.label}</span>
                    <input
                      className="w-full rounded-md border border-line bg-white px-2 py-1.5 text-xs outline-none focus:border-ink disabled:bg-slate-100 disabled:text-slate-400"
                      value={fieldRenames[field.path] ?? field.path}
                      onChange={(event) => onRename(field.path, event.target.value)}
                      disabled={!selectedFields[field.path]}
                      placeholder={field.path}
                    />
                  </label>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="border-t border-line bg-white p-3">
        <div className="mb-2 text-xs font-semibold uppercase tracking-normal text-slate-500">Generated Format</div>
        <pre className="max-h-36 overflow-auto rounded-md bg-slate-950 p-3 text-[11px] leading-5 text-slate-100">
          {JSON.stringify(configPreview, null, 2)}
        </pre>
      </div>
    </section>
  );
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [githubUrl, setGithubUrl] = useState("");
  const [linkedinUrl, setLinkedinUrl] = useState("");
  const [defaultRegion, setDefaultRegion] = useState("US");
  const [selectedFields, setSelectedFields] = useState(DEFAULT_SELECTED_FIELDS);
  const [keepEmptyFields, setKeepEmptyFields] = useState(DEFAULT_KEEP_EMPTY_FIELDS);
  const [fieldRenames, setFieldRenames] = useState(DEFAULT_FIELD_RENAMES);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const customConfig = useMemo(() => buildCustomConfig(selectedFields, fieldRenames, keepEmptyFields), [selectedFields, fieldRenames, keepEmptyFields]);

  function updateFiles(fileList) {
    const incoming = Array.from(fileList || []);
    const merged = [...files];
    const seen = new Set(files.map((file) => `${file.name}:${file.size}:${file.lastModified}`));
    for (const file of incoming) {
      const key = `${file.name}:${file.size}:${file.lastModified}`;
      if (!seen.has(key)) {
        merged.push(file);
        seen.add(key);
      }
    }
    if (merged.length > MAX_FILES) {
      setError(`Upload at most ${MAX_FILES} files.`);
      setFiles(merged.slice(0, MAX_FILES));
      return;
    }
    const invalid = merged.find((file) => {
      const lower = file.name.toLowerCase();
      return !ACCEPTED_EXTENSIONS.some((extension) => lower.endsWith(extension));
    });
    if (invalid) {
      setError(`${invalid.name} is not supported. Use CSV, JSON, TXT, MD, PDF, or DOCX.`);
      return;
    }
    const oversized = merged.find((file) => file.size > MAX_FILE_BYTES);
    if (oversized) {
      setError(`${oversized.name} exceeds the 10 MB file limit.`);
      return;
    }
    setError("");
    setFiles(merged);
  }

  function removeFile(fileToRemove) {
    setFiles((current) => current.filter((file) => file !== fileToRemove));
  }

  function toggleCustomField(path, checked) {
    setSelectedFields((current) => ({ ...current, [path]: checked }));
  }

  function toggleKeepEmptyField(path, checked) {
    setKeepEmptyFields((current) => ({ ...current, [path]: checked }));
  }

  function renameCustomField(path, value) {
    setFieldRenames((current) => ({ ...current, [path]: value }));
  }

  function selectAllCustomFields() {
    setSelectedFields(DEFAULT_SELECTED_FIELDS);
  }

  function clearAllCustomFields() {
    setSelectedFields(Object.fromEntries(CUSTOM_FIELD_OPTIONS.map((field) => [field.path, false])));
  }

  async function submit(event) {
    event.preventDefault();
    if (files.length > MAX_FILES) {
      setError(`Upload at most ${MAX_FILES} files.`);
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);

    const payload = new FormData();
    for (const file of files) payload.append("files", file);
    payload.append("github_url", githubUrl);
    payload.append("linkedin_url", linkedinUrl);
    payload.append("default_region", defaultRegion);
    payload.append("config", JSON.stringify(customConfig));

    try {
      const response = await fetch("/api/transform", {
        method: "POST",
        body: payload
      });
      const body = await response.json();
      if (!response.ok && body.error) throw new Error(body.error);
      setResult(body);
    } catch (err) {
      setError(err.message || "Transform failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen">
      <header className="border-b border-orange-100 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-5 py-5 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-normal">Multi-Source Candidate Transformer</h1>
            <p className="mt-1 text-sm text-slate-600">Canonical profile, provenance, confidence, and configurable output.</p>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-600">
            {result?.validation_errors?.length ? <XCircle className="text-coral" size={18} /> : <CheckCircle2 className="text-mint" size={18} />}
            {result ? `${result.validation_errors.length} validation issue(s)` : "Ready"}
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-5 px-5 py-5 lg:grid-cols-[420px_1fr]">
        <form onSubmit={submit} className="space-y-4 rounded-lg border border-orange-100 bg-white p-4 shadow-sm">
          <label className="block">
            <span className="mb-2 block text-sm font-medium">Sources</span>
            <div className="flex min-h-28 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-orange-300 bg-orange-50/70 px-4 py-5 text-center transition hover:border-orange-400 hover:bg-orange-100/70">
              <UploadCloud size={26} className="text-accent" />
              <span className="mt-2 max-w-full text-sm text-slate-700">CSV, JSON, TXT, MD, PDF, or DOCX</span>
              <span className="mt-1 text-xs text-slate-500">Max {MAX_FILES} files, 10 MB each</span>
              <input
                className="sr-only"
                type="file"
                multiple
                accept=".csv,.json,.txt,.md,.pdf,.docx"
                onChange={(event) => updateFiles(event.target.files)}
              />
            </div>
            {files.length > 0 && (
              <div className="mt-2 space-y-2">
                {files.map((file) => (
                  <div key={`${file.name}-${file.size}-${file.lastModified}`} className="flex items-center justify-between gap-2 rounded-md border border-line bg-white px-3 py-2">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium text-slate-800">{file.name}</div>
                      <div className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                    </div>
                    <button
                      type="button"
                      title="Remove file"
                      onClick={() => removeFile(file)}
                      className="rounded-md border border-line p-1.5 text-slate-500 hover:border-ink hover:text-ink"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </label>

          <label className="block">
            <span className="mb-2 flex items-center gap-2 text-sm font-medium">
              <Github size={16} />
              GitHub URL
            </span>
            <input
              className="w-full rounded-md border border-line px-3 py-2 text-sm outline-none focus:border-ink"
              value={githubUrl}
              onChange={(event) => setGithubUrl(event.target.value)}
              placeholder="https://github.com/username"
            />
          </label>

          <label className="block">
            <span className="mb-2 flex items-center gap-2 text-sm font-medium">
              <LinkIcon size={16} />
              LinkedIn URL
            </span>
            <input
              className="w-full rounded-md border border-line px-3 py-2 text-sm outline-none focus:border-ink"
              value={linkedinUrl}
              onChange={(event) => setLinkedinUrl(event.target.value)}
              placeholder="https://linkedin.com/in/username"
            />
          </label>

          <div className="grid gap-3">
            <label className="block">
              <span className="mb-2 block text-sm font-medium">Default Phone Region</span>
              <input
                className="w-full rounded-md border border-line px-3 py-2 text-sm uppercase outline-none focus:border-ink"
                value={defaultRegion}
                onChange={(event) => setDefaultRegion(event.target.value.toUpperCase())}
              />
              <span className="mt-1 block text-xs text-slate-500">Used only when a phone number has no country code.</span>
            </label>
          </div>

          <CustomFieldSelector
            selectedFields={selectedFields}
            keepEmptyFields={keepEmptyFields}
            fieldRenames={fieldRenames}
            onToggle={toggleCustomField}
            onToggleKeepEmpty={toggleKeepEmptyField}
            onRename={renameCustomField}
            onSelectAll={selectAllCustomFields}
            onClearAll={clearAllCustomFields}
            configPreview={customConfig}
          />

          <button
            type="submit"
            disabled={loading}
            title="Run transformer"
            className="flex w-full items-center justify-center gap-2 rounded-md bg-accent px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-accentDark disabled:cursor-wait disabled:opacity-70"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
            Transform
          </button>

          {error && <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>}
        </form>

        <div className="space-y-5">
          <JsonPanel title="Custom Output" data={result?.custom_output} />
          {result && <AgentOps diagnostics={result?.agent_diagnostics} />}
          <CleanProfile profile={result?.default_profile} />
          <div className="grid gap-5">
            <JsonPanel title="Canonical Profile" data={result?.default_profile} />
          </div>
        </div>
      </div>
    </main>
  );
}
