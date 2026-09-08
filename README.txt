rumik-oss-tts-eval
=================

how we measure text-to-speech quality for 21 indian languages, and how our
model compares to six commercial tts apis.

this repo has the prompts, the scoring code, the method, and our per-sample
numbers. it does not have audio: we cannot republish audio the vendor apis
generated for us, and our own is large and regenerable from the prompts.


what we found
-------------

character error rate (cer), lower is better. judged by indicconformer.
averaged over the 15 languages where at least one api competes:

  best api per language     0.0177   (the best of six, chosen per language)
  rumik-oss (ours)           0.0217

per language, against the best api available in that language:

  ahead          0
  within noise  12
  behind         3     (bengali, punjabi, urdu)

we cover 21 of 21 benchmark languages. elevenlabs covers 13, gemini 15,
sarvam 10, smallest 10, cartesia 9, grok 1. on the six languages no api
supports -- bodo, dogri, kashmiri, manipuri, sanskrit, santali -- there is
nothing to compare against.

against sarvam alone, on the 10 languages it supports, same 999 prompts each:
we win 550, lose 391, tie 58. the difference is in the tail -- at the 90th
percentile we are 0.0311 and sarvam is 0.1019.

charts: docs/vendor_bars.html and docs/sarvam_hist.html


two things to know before quoting any of this
---------------------------------------------

1. a gap smaller than 0.006 cer is noise. that is how much two runs of the
   same checkpoint differ. we mark those "within noise" instead of calling
   them wins. most of our per-language results are in that band.

2. on some languages our model scores better than real human speech does
   under this judge. indicconformer's median cer on real human bengali is
   0.0545; ours is 0.0302. where that happens the number is measuring the
   judge, not the speech, and cannot rank systems.

both are explained in METHODOLOGY.md. read it before drawing conclusions.


reproduce our numbers, no model needed
--------------------------------------

  pip install -e .
  pytest -q

270 tests, under a second, no gpu and no downloads. three more modules cover
audio-token parsing and need torch, so they skip unless you install the
[judges] extra -- 320 in total. they pin the metric definitions, including
the cases where cer implementations silently disagree
(does the denominator include spaces, grapheme clusters or code points,
what an empty hypothesis scores).

then rebuild our tables from the per-sample scores we published:

  python scripts/aggregate.py results/rumik_oss.per_sample.tsv \
      --vendors results/vendor_analysis.json

if you get our numbers, our arithmetic is right. if not, tell us.


generate our audio and score it
-------------------------------

  pip install -e ".[judges]"

the model is loaded from hugging face and nowhere else:

  https://huggingface.co/rumik-ai/rumik-oss-1

no weights are vendored into this repo. the mimi codec ships inside that
model repo, so there is nothing else to fetch.

  python scripts/generate.py --promptset prompts/indic22_bench.jsonl \
      --out audio/rumik_oss

  python scripts/score.py --audio audio/rumik_oss \
      --promptset prompts/indic22_bench.jsonl \
      --judge indicconformer --out runs/rumik_oss

  python scripts/aggregate.py runs/rumik_oss/per_sample.tsv \
      --vendors results/vendor_analysis.json

generation goes through the model's own generate_audio(), which consults its
stop predictor as well as the </audio> token. settings default to the published
ones: temperature 0.8, top_k 30, sampling restricted to the audio token ids,
and a per-prompt token budget derived from expected_seconds.

  verified: on 2026-09-08 this path was run from a clean clone against
  rumik-ai/rumik-oss-1 and reproduced the published per-prompt scores:

    prompt       published      live      duration ratio
    hi_000001      0.01205    0.01250        1.07
    hi_000002      0.02740    0.02516        1.19
    hi_000003      0.02532    0.02614        1.30
    hi_000004      0.00000    0.00529        1.09

  the residual differences are sampling noise -- generation is stochastic at
  temperature 0.8, so exact equality is not expected and would be suspicious.

  one thing to get right if you write your own generation script: the model is
  driven with the wire format

    <text>{speaker}: {text}<audio>

  and the `text` field in prompts/*.jsonl is only the middle part. the trailing
  <audio> is what puts the model into audio-emission mode. omit it and the model
  runs long -- we measured 1.3x to 2.5x the expected duration, and cer charges
  the trailing speech as insertions. scripts/generate.py builds this for you via
  bundle.prompt().


score any other system
----------------------

the judge is indicconformer (gated on hugging face -- request access, then
huggingface-cli login). generate audio for the prompts in prompts/, one file
per prompt id, then:

  python scripts/score.py --audio audio/my_system \
      --promptset prompts/indic22_bench.jsonl \
      --judge indicconformer --out runs/my_system

  python scripts/aggregate.py runs/my_system/per_sample.tsv \
      --vendors results/vendor_analysis.json

the output is the same schema as results/, so it lines up directly with ours.


how the vendor numbers were produced
------------------------------------

we called each vendor's own api with the same prompts in prompts/, following
that vendor's documented prompt format and using their own voices. we did not
try to make the apis behave like ours: each was driven the way its
documentation says to drive it.

every vendor was steered toward the same style target, "professional, steady
pace". how you express that differs per vendor, and this is a real limit on
any cross-vendor comparison -- some take words, some take numeric knobs, some
take a category:

  vendor                      model id                       voice     langs
  --------------------------------------------------------------------------
  rumik-oss (ours)             rumik-ai/rumik-oss-1           Ira          21
    style: description tag written inside the input text
  sarvam bulbul v3            bulbul:v3                      suhani       10
    style: numeric parameters (pace, temperature)
  elevenlabs v3               eleven_v3                      1Z7Y8o9c...  13
    style: numeric voice_settings
  smallest.ai lightning       lightning_v3.1_pro             aarini       10
    style: numeric parameter (speed)
  cartesia sonic 3.5          sonic-3.5                      per-language  9
    style: numeric parameters plus a categorical emotion
  gemini 3.1 flash tts        gemini-3.1-flash-tts-preview   Kore         15
    style: natural-language preamble in the prompt
  xai grok tts                unversioned                    eve           1
    style: numeric parameter (speed)

cartesia used a different voice per language. the full mapping, and every
field above, is in results/vendor_config.json so it is data, not prose.

our own model takes style inside the text, so the exact prompt is:

  <text>Ira: <description="professional, {lang} accent, steady pace"> {text}<audio>

the trailing <audio> is what puts it into audio mode. generation is limited
to the 16385 audio token ids. temperature 0.8, top_k 30.

to reproduce the vendor side you need your own account with each vendor.
the prompts and the exact configuration above are everything else you need.


what is in here
---------------

  README.txt              this file
  METHODOLOGY.md          how a system gets a number, and which numbers mean
                          anything. the noise floors are the important part.
  DATA_LICENSES.md        where every prompt came from and what its licence is
  LICENSE                 apache-2.0, for the code

  prompts/
    indic22_bench.jsonl   2100 prompts, 21 languages -- the benchmark set
                          behind every number above. generated by rumik ai.
    heldout_v1.jsonl       180 prompts, 18 languages -- never trained on.
                          130 of these are fleurs (cc-by-4.0, tagged `fleurs`).
    curated_hard_50.jsonl 1050 prompts, 21 languages -- deliberately hard,
                          every row stresses digit sequences.

  src/rumik_oss_eval/
    metrics/              cer, gcer, wer, length penalty, structural gates
    normalize/            markup strip, number expansion, iso-15919,
                          phonetic fold
    asr/                  judges: indicconformer, whisper, scribe, router
    rubric.py             gates plus weighted score composition

  scripts/aggregate.py    per-sample scores -> the published tables
  scripts/score.py        score a directory of audio against a prompt set
  scripts/check_publishable.py   fails if audio, weights or keys enter the tree

  results/                our per-sample scores, vendor scores, vendor config
                          note: the vendor files contain a column named
                          silk-oss-v2. that is an earlier rumik model the
                          published benchmark measured, not the model evaluated
                          here. it keeps its original name because those files
                          record what was measured.
  tests/                  320 tests (270 without torch)
  docs/                   the charts

prompt row format (one json object per line):

  id                stable key, also the audio filename and the row key in
                    results/*.per_sample.tsv
  language          iso code, bare -- the judge routes on this
  speaker           voice name for our model
  text              full wire-format prompt, description tag included
  spoken            what should be heard; this is the scoring reference
  expected_seconds  peer-median vendor duration, used by the length penalty
  tags              provenance and category


the model
---------

evaluated here: rumik-oss, loaded from https://huggingface.co/rumik-ai/rumik-oss-1
a lora (r=64, alpha=128, on the q/k/v/o projections) trained with grpo. only
this checkpoint is evaluated in this repo.

we want to be plain about one thing: the training run that produced it was
close to a null result. cer moved from 0.0241 to 0.0213 over 313 steps, and
the within-group reward spread (0.017) was below the judge's noise floor
(0.085), which means the training signal was largely ranking asr noise rather
than speech quality. the benchmark numbers above are real; the causal story
behind them is weak.


licence and attribution
-----------------------

the code in this repo is apache-2.0 (see LICENSE).

the model is not. rumik-oss is a fine-tune of CohereLabs/tiny-aya-fire, which
is cc-by-nc-4.0 -- attribution required, and commercial use prohibited. those
conditions carry to derivatives. the apache-2.0 grant here covers this code
and does not extend to the model weights.

other components we depend on and thank:

  mimi codec           kyutai            cc-by-4.0
  indicconformer 600m  ai4bharat         mit (access-gated on hf)
  whisper large-v3     openai            mit
  tiny-aya-fire        cohere labs       cc-by-nc-4.0

the prompts are ours: indic22_bench and curated_hard_50 were generated by
rumik ai for this benchmark. the tags on indic22_bench (wiki, book, news,
conv, alexa, bb, umang, proper_noun) are register labels -- the style a
prompt is written in -- not sources.

the exception is 130 rows in heldout_v1.jsonl, which come from fleurs and are
licensed cc-by-4.0:

  FLEURS: Few-shot Learning Evaluation of Universal Representations of
  Speech. conneau, ma, khanuja, zhang, axelrod, dalmia, riesa, rivera,
  bapna. ieee slt 2023.

those rows carry the `fleurs` tag, so you can filter them out if you need a
set free of third-party material. full detail in DATA_LICENSES.md.

full attribution is in NOTICE.

vendor names are used only to report measurements of publicly available
services. no affiliation or endorsement is implied, and no vendor-generated
audio is redistributed here.
