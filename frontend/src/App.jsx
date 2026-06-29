import { useMemo, useState } from "react";
import { CheckCircle2, FileJson, Github, Loader2, Play, UploadCloud, XCircle } from "lucide-react";

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

        <div className="grid gap-5 xl:grid-cols-2">
          <JsonPanel title="Canonical Profile" data={result?.default_profile} />
          <JsonPanel title="Custom Output" data={result?.custom_output} />
        </div>
      </div>
    </main>
  );
}

