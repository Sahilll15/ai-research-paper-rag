"use client";

import { useEffect, useRef, useState, type SyntheticEvent } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EXAMPLE_QUESTIONS = [
  "How does flash attention reduce memory usage?",
  "How does Constitutional AI differ from InstructGPT's RLHF?",
  "What is the key idea behind LoRA?",
];

const LOADING_STAGES = [
  "Searching the index…",
  "Weighing the retrieved passages…",
  "Drafting an answer…",
];

type SourceRef = {
  chunk_id: string;
  paper: string;
  page: number;
};

type QueryResponse = {
  answer: string;
  sources: SourceRef[];
  retries: number;
};

function retrievalNote(retries: number) {
  if (retries === 0) return "Found relevant context on the first search.";
  if (retries === 1)
    return "Took one extra search pass before the context was judged relevant.";
  return `Took ${retries} extra search passes before the context was judged relevant.`;
}

// The model occasionally appends its raw citation IDs to the answer text
// itself (e.g. "(Source Chunk IDs: [uuid], [uuid])") instead of keeping them
// only in the structured sources field. Strip that leak before it reaches the reader.
function sanitizeAnswer(text: string) {
  const uuid = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
  const trailingCitation = new RegExp(`\\s*\\([^()]*${uuid}[^()]*\\)\\.?\\s*$`, "i");
  return text.replace(trailingCitation, "").trim();
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [stage, setStage] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!loading) return;
    const timers = [
      setTimeout(() => setStage(1), 1200),
      setTimeout(() => setStage(2), 2800),
    ];
    return () => timers.forEach(clearTimeout);
  }, [loading]);

  async function runQuery(q: string) {
    if (!q.trim() || loading) return;

    setLoading(true);
    setStage(0);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      if (!res.ok) {
        throw new Error(`The index couldn't answer that (error ${res.status}). Try again in a moment.`);
      }
      const data: QueryResponse = await res.json();
      setResult(data);
    } catch (err) {
      const message =
        err instanceof TypeError
          ? "Can't reach the index right now. Check that the API server is running and try again."
          : err instanceof Error
            ? err.message
            : "Something went wrong answering that. Try again.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: SyntheticEvent<HTMLFormElement>) {
    e.preventDefault();
    runQuery(query);
  }

  function applyExample(q: string) {
    setQuery(q);
    inputRef.current?.focus();
  }

  return (
    <div className="flex min-h-screen justify-center bg-paper px-6 py-20">
      <main className="w-full max-w-[640px]">
        <header className="text-center">
          <h1 className="font-serif text-[2.25rem] italic leading-none text-ink">
            The Research Index
          </h1>
          <p className="mx-auto mt-4 max-w-[46ch] font-serif text-[1.05rem] leading-7 text-muted">
            A grounded question-answering index over forty-one AI research
            papers, from &ldquo;Attention Is All You Need&rdquo; to DeepSeek-R1.
          </p>
        </header>

        <div className="my-10 h-px bg-rule" />

        <form onSubmit={handleSubmit} className="flex items-end gap-4">
          <label className="flex-1">
            <span className="block font-sans text-[0.8rem] text-muted">
              Your question
            </span>
            <input
              ref={inputRef}
              className="mt-1 w-full border-b border-rule bg-transparent py-1.5 font-sans text-[0.95rem] text-ink outline-none placeholder:text-muted/70 focus:border-accent"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="How does flash attention reduce memory usage?"
            />
          </label>
          <button
            type="submit"
            disabled={loading}
            className="mb-1.5 shrink-0 rounded-sm font-sans text-[0.9rem] font-medium text-accent underline decoration-accent/40 underline-offset-4 hover:decoration-accent disabled:text-muted disabled:no-underline"
          >
            {loading ? "Searching…" : "Ask"}
          </button>
        </form>

        {error && (
          <p role="alert" className="mt-10 border-l-2 border-accent pl-4 font-sans text-[0.9rem] leading-6 text-ink">
            {error}
          </p>
        )}

        {!result && !error && !loading && (
          <div className="mt-10">
            <p className="font-sans text-[0.85rem] leading-6 text-muted">
              Try a question about a single paper, or one that compares two
              approaches — the index searches across all of them at once. A
              few to start with:
            </p>
            <ul className="mt-5 space-y-2.5">
              {EXAMPLE_QUESTIONS.map((q) => (
                <li key={q}>
                  <button
                    type="button"
                    onClick={() => applyExample(q)}
                    className="rounded-sm text-left font-serif text-[0.95rem] italic leading-6 text-ink/80 underline decoration-rule underline-offset-4 hover:text-ink hover:decoration-accent"
                  >
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {loading && (
          <div className="mt-10">
            <p aria-live="polite" className="font-sans text-[0.85rem] leading-6 text-muted">
              {LOADING_STAGES[stage]}
            </p>
            <div className="relative mt-4 h-px w-full overflow-hidden bg-rule" aria-hidden="true">
              <div className="absolute inset-y-0 w-1/5 bg-accent animate-[scan_1.6s_ease-in-out_infinite]" />
            </div>
          </div>
        )}

        {result && (
          <div className="mt-10 animate-[reveal_0.5s_ease-out]">
            <p className="font-serif text-[1.15rem] leading-8 text-ink">
              {sanitizeAnswer(result.answer)}
            </p>

            {result.sources.length > 0 && (
              <>
                <ol className="mt-8 space-y-2 border-t border-rule pt-6">
                  {result.sources.map((s, i) => (
                    <li
                      key={s.chunk_id}
                      className="flex gap-3 font-sans text-[0.85rem] leading-6 text-muted"
                    >
                      <span className="text-accent">{i + 1}.</span>
                      <span>
                        {s.paper}
                        <span className="text-muted/70"> — p.{s.page + 1}</span>
                      </span>
                    </li>
                  ))}
                </ol>

                <details className="mt-6 border-t border-rule pt-4">
                  <summary className="cursor-pointer select-none font-sans text-[0.75rem] text-muted hover:text-ink">
                    Retrieval detail
                  </summary>
                  <div className="mt-3 space-y-2 font-sans text-[0.75rem] leading-6 text-muted/80">
                    <p>{retrievalNote(result.retries)}</p>
                    <ul className="space-y-1 font-mono text-[0.7rem]">
                      {result.sources.map((s) => (
                        <li key={s.chunk_id}>{s.chunk_id}</li>
                      ))}
                    </ul>
                  </div>
                </details>
              </>
            )}
          </div>
        )}
      </main>

      <style>{`
        @keyframes reveal {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes scan {
          0% { left: -20%; }
          100% { left: 100%; }
        }
      `}</style>
    </div>
  );
}
