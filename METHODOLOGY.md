# Judging methodology

How a TTS system gets a number here, and which of those numbers mean anything.

## The chain

```
prompt text ──▶ TTS system ──▶ audio ──▶ ASR judge ──▶ hypothesis text
                                                          │
reference text ───────────────────────────────────────────┤
                                                          ▼
                                            normalise both, then CER / WER
```

Every step is a place a number can be wrong for reasons unrelated to speech
quality. The rest of this document is about those places.

## The judge

**IndicConformer** (`ai4bharat/indic-conformer-600m-multilingual`, RNNT decoder)
is the neutral referee. It was chosen because it is the only judge in our
comparison with complete coverage, 21 of 21 languages. Whisper large-v3 covers
13, SraVaani 19.

Using a single judge across all systems is the point: **absolute CER is a
property of the judge as much as of the speech**, so numbers are only comparable
when the judge is held fixed. Cross-judge comparisons are not made anywhere in
this repo.

A judge that shares a vendor with the system under test is excluded from that
system's score, a vendor's own ASR tends to flatter its own TTS.

## Score tiers

Each hypothesis is scored three ways; **`normalized` is the tier used for every
published number** here.

| Tier | What it does |
|---|---|
| `strict` | raw text, no normalisation |
| **`normalized`** | markup stripped, romanised to ISO-15919, phonetic fold applied |
| `llm_equivalence` | additionally forgives spelling variants judged equivalent by an LLM |

`llm_equivalence` is recorded but not used: it introduces a second model into the
measurement, and its forgiveness rate can differ per language and per system.

## Normalisation

`src/rumik_oss_eval/normalize/`, applied identically to reference and hypothesis:

1. **markup**, speaker prefix, `<description="…">`, tone tags, inline events
   removed. The model is not credited or penalised for markup it correctly did
   not voice.
2. **numbers**, digit strings expanded to words. See the ITN caveat below.
3. **transliteration**, ISO-15919 romanisation, so scripts are compared on a
   common surface.
4. **fold**, phonetic folding of variants that differ orthographically but not
   in pronunciation.

## Metrics

- **CER**, character edit distance ÷ reference length, spaces stripped first.
- **gCER**, the same over **grapheme clusters**. `cer` splits Unicode code
  points, which understates error on every Indic script here and by a *different
  factor per script*, so gCER is the one to trust for cross-language comparison.
- **WER**, word edit distance ÷ reference word count.
- **length**, an asymmetric dead-zone penalty on generated ÷ expected duration.
  Inside `[0.95, 1.25]` the penalty is zero.

All are clamped to `[0, 1]` and never return NaN. That matters when a metric is
used as a reward: an unbounded or NaN term poisons any aggregate that contains
it, and a NaN that gets filtered out of a mean silently makes "produce nothing"
the optimal strategy.

## Gates

A generation scores **0**, not a CER, when it fails structurally:

- no `</audio>` emitted, or no decodable frame
- the ASR judge returned a failure status
- the reference is missing (marked invalid and excluded from means, rather than
  scored zero, since that is a corpus bug and not a model failure)

Gated samples are counted and reported separately. Folding them into a CER mean
would conflate "said it wrong" with "produced nothing".

## The noise floor, the most important section

**A CER difference smaller than the measurement noise is not a result.**

Two sources of noise are measured rather than assumed:

**Run-to-run.** Two runs of the *same checkpoint* differ by about **0.006 CER**
per language. Any per-language gap below that is reported as "within noise".

**The judge's own floor.** Running *real human speech* through the same judge and
the same normaliser gives a non-zero CER, the error the apparatus contributes by
itself. On our corpus that floor is **0.085 mean / 0.070 median** on Hindi
(hinglish60k), and it varies substantially by language: IndicConformer's median
CER on real human Bengali is **0.0545**, on Punjabi **0.0719**.

The consequence is sharp: **on several languages our TTS scores better than real
human speech does under this judge.** Where that happens, CER is measuring judge
noise and cannot rank systems. Check the floor for a language before drawing a
conclusion from a small margin in it.

## Known confounds

**No inverse text normalisation.** None of the ASR judges perform ITN, so a
digit-bearing reference is unmatchable by construction, the judge emits words,
the reference has digits, and the normaliser's expansion may not agree with the
judge's phrasing. Any prompt set heavy in digits (ours: `curated_hard_50 stress_digits`)
is confounded, and improvements on it should not be read as intelligibility
gains.

**Coverage is not comparability.** Vendors support different language subsets. A
blank cell means "no support", never "scored badly", and a mean over a vendor's
own languages is not comparable to a mean over another's. Published tables carry
both: each column over its own coverage, and every column restricted to a shared
language set.

**Sample size.** Per-language n is 100 unless stated. A single language's CER at
n=100 carries meaningful variance; the aggregate over 21 languages is much
steadier than any one row.

## Reproducing a number

```bash
pip install -e .
pytest -q                      # 270 tests; 320 with the [judges] extra

python scripts/generate.py --promptset prompts/indic22_bench.jsonl --out audio/rumik_oss
python scripts/score.py    --audio audio/rumik_oss \
                           --promptset prompts/indic22_bench.jsonl \
                           --judge indicconformer --out runs/rumik_oss
python scripts/aggregate.py runs/rumik_oss/per_sample.tsv
```

The model is loaded from `rumik-ai/rumik-oss-1` on the Hugging Face Hub; no
weights are vendored here.

`results/` holds our published per-sample scores. To check our arithmetic without
running any model, recompute the aggregates from those files, the per-sample TSVs
are the same rows the tables were built from.
