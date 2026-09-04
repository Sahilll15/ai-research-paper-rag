import sys
import types

# ragas 0.4.3 unconditionally imports a class langchain-community no longer ships.
# We never use Vertex AI, so stub it out rather than downgrade a working dependency.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class _ChatVertexAI:
    pass


_stub.ChatVertexAI = _ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _stub

import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate

# NOTE: import from ragas.metrics (legacy), not ragas.metrics.collections.
# The deprecation warning tells you to switch to .collections, but evaluate()
# in this ragas version only accepts the legacy Metric base class - passing
# .collections instances raises "All metrics must be initialised metric objects".
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from src.rag.pipeline import rag_graph

load_dotenv()

_judge_llm = ChatOpenAI(model="gpt-4o-mini")
_judge_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def load_dataset(path: str = "eval/dataset.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(dataset_path: str = "eval/dataset.jsonl"):
    entries = load_dataset(dataset_path)
    samples = []

    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry['question']}")

        state = rag_graph.invoke({"query": entry["question"]})
        retrieved_contexts = [doc.page_content for doc in state["chunks"]]

        samples.append(
            SingleTurnSample(
                user_input=entry["question"],
                response=state["answer"],
                retrieved_contexts=retrieved_contexts,
                reference=entry["ground_truth"],
            )
        )

    dataset = EvaluationDataset(samples=samples)

    results = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=_judge_llm,
        embeddings=_judge_embeddings,
    )

    print(results)

    os.makedirs("eval/runs", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out_path = f"eval/runs/{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results.to_pandas().to_dict(orient="records"), f, indent=2)
    print(f"Saved per-question results to {out_path}")

    return results


if __name__ == "__main__":
    run_eval()
