import { useMemo, useState } from "react";
import {
  BriefcaseBusiness,
  CheckCircle2,
  FileJson,
  Github,
  GraduationCap,
  Link as LinkIcon,
  Loader2,
  Mail,
  MapPin,
  Phone,
  Play,
  Sparkles,
  UploadCloud,
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
    ([name]) => !["Education", "Experience", "Skills", "Skills Summary"].includes(name)
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
      </div>
    </section>
  );
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [githubUrl, setGithubUrl] = useState("");
  const [defaultRegion, setDefaultRegion] = useState("US");
  const [config, setConfig] = useState(SAMPLE_CONFIG);
  const [useLlm, setUseLlm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const fileNames = useMemo(() => files.map((file) => file.name).join(", "), [files]);

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const payload = new FormData();
    for (const file of files) payload.append("files", file);
    payload.append("github_url", githubUrl);
    payload.append("default_region", defaultRegion);
    payload.append("use_llm", String(useLlm));
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
              <span className="mt-2 max-w-full text-sm text-slate-700">{fileNames || "CSV, JSON, or TXT"}</span>
              <input
                className="sr-only"
                type="file"
                multiple
                accept=".csv,.json,.txt,.md"
                onChange={(event) => setFiles(Array.from(event.target.files || []))}
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
