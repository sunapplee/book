# Метрики RAG. Источник: sibur_preparing/notebooks/rag_metrics_full.ipynb
# Поиск: Precision/Recall@k, HitRate, MRR, MAP, NDCG.
# Эталон: EM, token-F1, ROUGE, BLEU, chrF. Судья: Ollama.
# Для судьи: pip install langchain-ollama; ollama pull qwen3.5:4b
# Лексические метрики не проверяют факты; RAGAS — в llama_cpp/llama_cpp_rag.py.

from __future__ import annotations
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean


# 1. Формат данных и два примера из исходного набора
@dataclass
class Sample:
    question: str
    answer: str                                         # что сгенерировал RAG
    contexts: list[str] = field(default_factory=list)   # тексты топ-k чанков

    retrieved_ids: list[str] = field(default_factory=list)   # id в ПОРЯДКЕ выдачи
    relevant_ids: list[str] = field(default_factory=list)    # разметка релевантных
    relevance_grades: dict = field(default_factory=dict)     # для NDCG: id -> 0..3

    reference: str | None = None

    meta: dict = field(default_factory=dict)

def load_samples(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [Sample(**r) for r in rows]

SAMPLES = [
    Sample(
        question="До какого числа действует договор аренды?",
        # "продлевается автоматически" в контексте НЕТ -> галлюцинация
        answer="Договор аренды действует до 31 декабря 2026 года и продлевается автоматически.",
        contexts=["Пункт 4.1. Договор аренды заключен на срок до 31 декабря 2026 года.",
                  "Пункт 9.2. Стороны уведомляют друг друга письменно."],
        retrieved_ids=["c41", "c92"], relevant_ids=["c41"],
        reference="Договор действует до 31 декабря 2026 года.",
        meta={"type": "factoid"}),
    Sample(
        question="Какая ставка по депозиту и есть ли капитализация?",
        answer="Ставка по депозиту составляет 7,5% годовых.",   # ответ НЕПОЛНЫЙ
        contexts=["Ставка по вкладу Накопительный - 7,5% годовых.",
                  "Капитализация процентов производится ежемесячно.",
                  "Минимальная сумма вклада 50 000 рублей."],
        retrieved_ids=["d10", "d11", "d12"], relevant_ids=["d10", "d11"],
        reference="Ставка 7,5% годовых, капитализация ежемесячная.",
        meta={"type": "multi_hop"}),
]

# 2. Качество поиска: нужны ID найденных и релевантных чанков
def precision_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / k if k > 0 else 0.0

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return float("nan")          # нечего искать — пример не учитываем
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def hit_rate_at_k(retrieved, relevant, k):
    return float("nan") if not relevant else float(bool(set(retrieved[:k]) & relevant))

def reciprocal_rank(retrieved, relevant):
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0

def average_precision(retrieved, relevant):
    # В исходнике делитель ограничен длиной выдачи: при короткой выдаче AP завышается.
    if not relevant:
        return float("nan")
    hits = score = 0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            score += hits / i                  # precision@i в момент попадания
    return score / min(len(relevant), len(retrieved)) if hits else 0.0

def ndcg_at_k(retrieved, grades, k):
    dcg = sum(grades.get(d, 0.0) / math.log2(i + 1)
              for i, d in enumerate(retrieved[:k], start=1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return float("nan") if idcg == 0 else dcg / idcg

def nanmean(xs):
    vals = [x for x in xs if x == x]           # NaN != NaN
    return mean(vals) if vals else float("nan")

def evaluate_retrieval(samples, ks=(1, 3, 5, 10)):
    acc = {}
    for s in samples:
        relevant = set(s.relevant_ids)
        if not relevant:
            continue
        grades = s.relevance_grades or {d: 1.0 for d in relevant}
        for k in ks:
            acc.setdefault(f"precision@{k}", []).append(precision_at_k(s.retrieved_ids, relevant, k))
            acc.setdefault(f"recall@{k}", []).append(recall_at_k(s.retrieved_ids, relevant, k))
            acc.setdefault(f"hit_rate@{k}", []).append(hit_rate_at_k(s.retrieved_ids, relevant, k))
            acc.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(s.retrieved_ids, grades, k))
        acc.setdefault("mrr", []).append(reciprocal_rank(s.retrieved_ids, relevant))
        acc.setdefault("map", []).append(average_precision(s.retrieved_ids, relevant))
    return {name: nanmean(v) for name, v in acc.items()}

# 3. Сравнение ответа с эталоном
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_ARTICLES = {"a", "an", "the"}

def normalize(text, drop_articles=True):
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    tokens = _PUNCT.sub(" ", text).split()
    return " ".join(t for t in tokens if not (drop_articles and t in _ARTICLES))

def tokenize(text):
    return normalize(text).split()

def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def exact_match(pred, ref):
    return float(normalize(pred) == normalize(ref))

def token_f1(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return float(p == r)
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)

def rouge_n(pred, ref, n=1, mode="f1"):
    p, r = ngrams(tokenize(pred), n), ngrams(tokenize(ref), n)
    if not p or not r:
        return 0.0
    overlap = sum((p & r).values())
    recall, precision = overlap / sum(r.values()), overlap / sum(p.values())
    if mode == "recall":
        return recall
    if mode == "precision":
        return precision
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

def _lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]

def rouge_l(pred, ref, beta=1.2):
    p, r = tokenize(pred), tokenize(ref)
    lcs = _lcs_len(p, r) if p and r else 0
    if lcs == 0:
        return 0.0
    precision, recall, b2 = lcs / len(p), lcs / len(r), beta ** 2
    return (1 + b2) * precision * recall / (recall + b2 * precision)

def bleu(pred, ref, max_n=4, smooth=True):
    p_toks, r_toks = tokenize(pred), tokenize(ref)
    if not p_toks or not r_toks:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        p_ng, r_ng = ngrams(p_toks, n), ngrams(r_toks, n)
        if not p_ng:
            logs.append(math.log(1e-9))
            continue
        clipped = sum(min(c, r_ng[ng]) for ng, c in p_ng.items())
        total = sum(p_ng.values())
        if smooth:                                  # smoothing method 1
            clipped, total = clipped + 1, total + 1
        logs.append(math.log(clipped / total) if clipped else math.log(1e-9))
    bp = 1.0 if len(p_toks) > len(r_toks) else math.exp(1 - len(r_toks) / len(p_toks))
    return bp * math.exp(sum(logs) / max_n)

def chrf(pred, ref, max_n=6, beta=2.0):
    p_txt, r_txt = normalize(pred).replace(" ", ""), normalize(ref).replace(" ", "")
    if not p_txt or not r_txt:
        return 0.0
    scores, b2 = [], beta ** 2
    for n in range(1, max_n + 1):
        p_ng = Counter(p_txt[i:i + n] for i in range(len(p_txt) - n + 1))
        r_ng = Counter(r_txt[i:i + n] for i in range(len(r_txt) - n + 1))
        if not p_ng or not r_ng:
            continue
        overlap = sum((p_ng & r_ng).values())
        if overlap == 0:
            scores.append(0.0)
            continue
        precision, recall = overlap / sum(p_ng.values()), overlap / sum(r_ng.values())
        scores.append((1 + b2) * precision * recall / (b2 * precision + recall))
    return mean(scores) if scores else 0.0

def evaluate_reference(samples):
    rows = [{
        "exact_match": exact_match(s.answer, s.reference),
        "token_f1": token_f1(s.answer, s.reference),
        "rouge1_f1": rouge_n(s.answer, s.reference, 1),
        "rouge1_recall": rouge_n(s.answer, s.reference, 1, mode="recall"),
        "rouge2_f1": rouge_n(s.answer, s.reference, 2),
        "rougeL": rouge_l(s.answer, s.reference),
        "bleu": bleu(s.answer, s.reference),
        "bleu2": bleu(s.answer, s.reference, max_n=2),
        "chrf": chrf(s.answer, s.reference),
        "len_ratio": len(tokenize(s.answer)) / max(len(tokenize(s.reference)), 1),
    } for s in samples if s.reference]
    return {k: mean(r[k] for r in rows) for k in rows[0]} if rows else {}

# 4. LLM-судья: подтверждение фактов, полезность контекста, полнота ответа
PROMPT_VERSION = "v1"          # логировать в отчёт вместе с метриками

SYS = "Ты строгий оценщик качества RAG-систем. Отвечаешь только валидным JSON."

CLAIMS_PROMPT = """Разбей ответ на атомарные утверждения — минимальные самостоятельные факты.
Раскрой местоимения, чтобы каждое было понятно вне контекста.
Вводные фразы и оценки не включай.

Ответ:
{answer}

JSON: {{"claims": ["...", "..."]}}"""

VERIFY_PROMPT = """Следует ли утверждение из контекста?

Контекст:
{context}

Утверждение: {claim}

Правила:
- "yes"     — прямо следует из контекста или тривиально из него выводится;
- "no"      — противоречит контексту либо добавляет отсутствующие в нём факты;
- "unclear" — контекст не позволяет решить.
Общеизвестные факты нарушением не считаются.

JSON: {{"reason": "одно предложение", "verdict": "yes|no|unclear"}}"""

CTX_PROMPT = """Какие фрагменты содержат информацию, полезную для ответа на вопрос?

Вопрос: {question}

Фрагменты:
{numbered}

JSON: {{"reason": "одно предложение", "useful": [номера]}}"""

RELEVANCY_PROMPT = """Отвечает ли ответ на заданный вопрос по существу?
Длина ответа достоинством не является. Уход в смежную тему — не отвечает.

Вопрос: {question}
Ответ: {answer}

JSON: {{"reason": "одно предложение", "relevant": true|false}}"""

COMPLETENESS_PROMPT = """Разбей вопрос на подвопросы и проверь, отвечен ли каждый.

Вопрос: {question}
Ответ: {answer}

JSON: {{"subquestions": [{{"q": "...", "answered": true|false}}]}}"""

PAIRWISE_PROMPT = """Какой из двух ответов лучше отвечает на вопрос с опорой на контекст?
Длина сама по себе достоинством не является. Порядок предъявления не значим.

Вопрос: {question}

Контекст:
{context}

Ответ A:
{a}

Ответ B:
{b}

JSON: {{"reason": "одно предложение", "winner": "A|B|tie"}}"""

def ollama_judge(model="qwen3.5:4b", base_url="http://localhost:11434"):
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=model, base_url=base_url,
                     temperature=0, reasoning=False, format="json")
    def call(system, user):
        r = llm.invoke([{"role": "system", "content": system},
                        {"role": "user", "content": user}])
        return r.content
    return call

def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group()) if m else {}

class LLMJudge:
    def __init__(self, call_fn, join="\n\n---\n\n"):
        self.call, self.join = call_fn, join
        self._cache = {}

    def _ask(self, user):
        if user not in self._cache:
            self._cache[user] = self.call(SYS, user)
        return parse_json(self._cache[user])

    def faithfulness(self, answer, contexts):
        ctx = self.join.join(contexts)
        claims = self._ask(CLAIMS_PROMPT.format(answer=answer)).get("claims", [])
        if not claims:
            return {"faithfulness": float("nan"), "claims": []}
        verdicts = [self._ask(VERIFY_PROMPT.format(context=ctx, claim=c)) for c in claims]
        supported = [v.get("verdict") == "yes" for v in verdicts]
        return {
            "faithfulness": mean(supported),
            "claims": [{"claim": c, "verdict": v.get("verdict"), "reason": v.get("reason")}
                       for c, v in zip(claims, verdicts)],
        }

    def context_relevance(self, question, contexts):
        if not contexts:
            return {"context_relevance": float("nan")}
        numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        useful = self._ask(CTX_PROMPT.format(question=question, numbered=numbered)).get("useful", [])
        return {"context_relevance": len([u for u in useful if isinstance(u, int)]) / len(contexts)}

    def answer_relevancy(self, question, answer):
        r = self._ask(RELEVANCY_PROMPT.format(question=question, answer=answer))
        return {"answer_relevancy": float(bool(r.get("relevant")))}

    def completeness(self, question, answer):
        subs = self._ask(COMPLETENESS_PROMPT.format(question=question,
                                                    answer=answer)).get("subquestions", [])
        return {"completeness": mean(bool(s.get("answered")) for s in subs) if subs else float("nan")}

    def pairwise(self, question, contexts, a, b):
        ctx = self.join.join(contexts)
        fwd = self._ask(PAIRWISE_PROMPT.format(question=question, context=ctx, a=a, b=b))
        rev = self._ask(PAIRWISE_PROMPT.format(question=question, context=ctx, a=b, b=a))
        flip = {"A": "B", "B": "A", "tie": "tie"}
        w1, w2 = fwd.get("winner"), rev.get("winner")
        return w1 if w1 == flip.get(w2) else "tie"

    def evaluate(self, samples):
        details = []
        for s in samples:
            row = {"question": s.question}
            row.update(self.faithfulness(s.answer, s.contexts))
            row.update(self.context_relevance(s.question, s.contexts))
            row.update(self.answer_relevancy(s.question, s.answer))
            row.update(self.completeness(s.question, s.answer))
            details.append(row)
        keys = ["faithfulness", "context_relevance", "answer_relevancy", "completeness"]
        scores = {k: nanmean([d.get(k, float("nan")) for d in details]) for k in keys}
        scores["prompt_version"] = PROMPT_VERSION
        return scores, details

# 5. Применение (оценки поиска и эталона не требуют LLM)
print(evaluate_retrieval(SAMPLES))
print(evaluate_reference(SAMPLES))

# Запускать с работающей Ollama:
# judge = LLMJudge(ollama_judge())
# scores, details = judge.evaluate(SAMPLES)
# print(scores)
# s = SAMPLES[0]
# print(judge.pairwise(s.question, s.contexts, s.answer, s.reference))
