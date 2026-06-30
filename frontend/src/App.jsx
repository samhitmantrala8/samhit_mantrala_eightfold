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

const SAMPLE_CONFIG = `{
  "fields": [
    { "path": "full_name", "type": "string", "required": true },
    { "path": "primary_email", "from": "emails[0]", "type": "string", "required": true },
    { "path": "phone", "from": "phones[0]", "type": "string", "normalize": "E164" },
    { "path": "github", "from": "links.github", "type": "string" },
    { "path": "skills", "from": "skills[].name", "type": "string[]", "normalize": "canonical" }
  ],
  "include_confidence": true,
  "include_provenance": false,
  "on_missing": "null"
}`;

const MAX_FILES = 5;
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = [".csv", ".json", ".txt", ".md", ".pdf"];

function JsonPanel({ title, data }) {
  return (
    <section className="rounded-lg border border-line bg-white">
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
        <div className="grid grid-cols-[1.1fr_1.4fr_1.2fr_80px] border-b border-line bg-slate-50 px-3 py-2 text-xs font-semibold uppercase tracking-normal text-slate-500">
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
        <div className="grid grid-cols-[1.2fr_0.8fr_1fr_70px_70px] border-b border-line bg-slate-50 px-3 py-2 text-xs font-semibold uppercase tracking-normal text-slate-500">
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

function AgentOps({ profile }) {
  const [selectedTrace, setSelectedTrace] = useState(null);
  const llmops = profile?.llmops;
  if (!llmops) return null;
  const iterations = llmops.iterations || [];
  const events = llmops.request_events || [];
  const taskTraces = llmops.task_traces || [];
  const finalScore = Number(llmops.final_score || 0);
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
        <Gauge size={16} />
        Agent Evaluation
      </div>
      <div className="space-y-3 rounded-md border border-line bg-slate-50 p-3">
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
    </div>
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

        <div>
          <div className="mb-2 text-sm font-semibold">Skills</div>
          <SkillChips skills={profile.skills || []} />
        </div>

        <MergeTrust profile={profile} />
        <SemanticMappings profile={profile} />
        <AgentOps profile={profile} />
      </div>
    </section>
  );
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [githubUrl, setGithubUrl] = useState("");
  const [linkedinUrl, setLinkedinUrl] = useState("");
  const [defaultRegion, setDefaultRegion] = useState("US");
  const [config, setConfig] = useState(SAMPLE_CONFIG);
  const [useLlm, setUseLlm] = useState(false);
  const [useGeminiHybrid, setUseGeminiHybrid] = useState(false);
  const [useAgenticLlmpops, setUseAgenticLlmpops] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const fileNames = useMemo(() => files.map((file) => file.name).join(", "), [files]);

  function updateFiles(fileList) {
    const selected = Array.from(fileList || []);
    if (selected.length > MAX_FILES) {
      setError(`Upload at most ${MAX_FILES} files.`);
      setFiles(selected.slice(0, MAX_FILES));
      return;
    }
    const invalid = selected.find((file) => {
      const lower = file.name.toLowerCase();
      return !ACCEPTED_EXTENSIONS.some((extension) => lower.endsWith(extension));
    });
    if (invalid) {
      setError(`${invalid.name} is not supported. Use CSV, JSON, TXT, MD, or PDF.`);
      return;
    }
    const oversized = selected.find((file) => file.size > MAX_FILE_BYTES);
    if (oversized) {
      setError(`${oversized.name} exceeds the 10 MB file limit.`);
      return;
    }
    setError("");
    setFiles(selected);
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
    payload.append("use_llm", String(useLlm));
    payload.append("use_gemini_hybrid", String(useGeminiHybrid));
    payload.append("use_agentic_llmops", String(useAgenticLlmpops));
    payload.append("config", config);

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
      <header className="border-b border-line bg-white">
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
        <form onSubmit={submit} className="space-y-4 rounded-lg border border-line bg-white p-4">
          <label className="block">
            <span className="mb-2 block text-sm font-medium">Sources</span>
            <div className="flex min-h-28 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-5 text-center">
              <UploadCloud size={26} className="text-slate-500" />
              <span className="mt-2 max-w-full text-sm text-slate-700">{fileNames || "CSV, JSON, TXT, MD, or PDF"}</span>
              <span className="mt-1 text-xs text-slate-500">Max {MAX_FILES} files, 10 MB each</span>
              <input
                className="sr-only"
                type="file"
                multiple
                accept=".csv,.json,.txt,.md,.pdf"
                onChange={(event) => updateFiles(event.target.files)}
              />
            </div>
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

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="mb-2 block text-sm font-medium">Phone Region</span>
              <input
                className="w-full rounded-md border border-line px-3 py-2 text-sm uppercase outline-none focus:border-ink"
                value={defaultRegion}
                onChange={(event) => setDefaultRegion(event.target.value.toUpperCase())}
              />
            </label>
            <label className="flex items-end gap-2 rounded-md border border-line px-3 py-2 text-sm">
              <input type="checkbox" checked={useLlm} onChange={(event) => setUseLlm(event.target.checked)} />
              LLM extractor
            </label>
            <label className="flex items-center gap-2 rounded-md border border-line px-3 py-2 text-sm">
              <input type="checkbox" checked={useGeminiHybrid} onChange={(event) => setUseGeminiHybrid(event.target.checked)} />
              Gemini hybrid
            </label>
            <label className="flex items-center gap-2 rounded-md border border-line px-3 py-2 text-sm">
              <input type="checkbox" checked={useAgenticLlmpops} onChange={(event) => setUseAgenticLlmpops(event.target.checked)} />
              Agent evaluator
            </label>
          </div>

          <label className="block">
            <span className="mb-2 block text-sm font-medium">Custom Config</span>
            <textarea
              className="h-72 w-full resize-y rounded-md border border-line bg-slate-50 px-3 py-2 font-mono text-xs leading-5 outline-none focus:border-ink"
              value={config}
              onChange={(event) => setConfig(event.target.value)}
              spellCheck="false"
            />
          </label>

          <button
            type="submit"
            disabled={loading}
            title="Run transformer"
            className="flex w-full items-center justify-center gap-2 rounded-md bg-ink px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-wait disabled:opacity-70"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
            Transform
          </button>

          {error && <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>}
        </form>

        <div className="space-y-5">
          <CleanProfile profile={result?.default_profile} />
          <div className="grid gap-5 xl:grid-cols-2">
            <JsonPanel title="Canonical Profile" data={result?.default_profile} />
            <JsonPanel title="Custom Output" data={result?.custom_output} />
          </div>
        </div>
      </div>
    </main>
  );
}
