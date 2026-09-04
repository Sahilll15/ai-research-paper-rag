"use client";

import { useState, type SyntheticEvent } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  async function handleSubmit(e: SyntheticEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) throw new Error(`The index couldn't answer that (${res.status}).`);
      const data: QueryResponse = await res.json();
      setResult(data);
    } catch {
      setError("The index couldn't answer that. Check that the API is running and try again.");
    } finally {
      setLoading(false);
    }
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
              className="mt-1 w-full border-b border-rule bg-transparent py-1.5 font-sans text-[0.95rem] text-ink outline-none placeholder:text-muted/70 focus:border-accent"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="How does flash attention reduce memory usage?"
            />
          </label>
          <button
            type="submit"
            disabled={loading}
            className="mb-1.5 shrink-0 font-sans text-[0.9rem] font-medium text-accent underline decoration-accent/40 underline-offset-4 hover:decoration-accent disabled:text-muted disabled:no-underline"
          >
            {loading ? "Searching…" : "Ask"}
          </button>
        </form>

        {error && (
          <p className="mt-10 border-l-2 border-accent pl-4 font-sans text-[0.9rem] leading-6 text-ink">
            {error}
          </p>
        )}

        {!result && !error && !loading && (
          <p className="mt-10 font-sans text-[0.85rem] leading-6 text-muted">
            Try a question about a single paper, or one that compares two
            approaches — the index searches across all of them at once.
          </p>
        )}

        {result && (
          <div className="mt-10 animate-[reveal_0.5s_ease-out]">
            <p className="font-serif text-[1.15rem] leading-8 text-ink">
              {result.answer}
            </p>

            {result.sources.length > 0 && (
              <ol className="mt-8 space-y-2 border-t border-rule pt-6">
                {result.sources.map((s, i) => (
                  <li
                    key={s.chunk_id}
                    className="flex gap-3 font-sans text-[0.85rem] leading-6 text-muted"
                  >
                    <span className="text-accent">{i + 1}.</span>
                    <span>
                      {s.paper}
                      <span className="text-muted/70"> — p.{s.page + 1}</span>{" "}
                      <span className="font-mono text-[0.75rem] text-muted/70">
                        [{s.chunk_id.slice(0, 8)}]
                      </span>
                    </span>
                  </li>
                ))}
              </ol>
            )}

            <p className="mt-6 font-sans text-[0.75rem] text-muted/70">
              {result.retries} retrieval{result.retries === 1 ? "" : "s"}
            </p>
          </div>
        )}
      </main>

      <style>{`
        @keyframes reveal {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
