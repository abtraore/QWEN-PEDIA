# qwen-pedia agent instructions

Everything in this repo is outward-facing: READMEs, recipe notes, commit
messages, PR bodies. The writing rules below are mandatory.

## Repo conventions

- **Every recipe directory carries its own README.md** with, in this order:
  headline measured numbers, a contents list (one line per file), and a
  ready-to-copy `docker run` command (build step first when the recipe has
  a Dockerfile). End with the endpoint URL, the served model name, and any
  first-use warning (cold start, load time, host RAM needs).
- Recipe directory layout: `README.md` (copy-paste path), `launch.sh`
  (parameterized launcher), `NOTES.md` (the whys: traps, knob sweeps,
  operational rules), plus `Dockerfile` + `patches/` only when upstream
  images are not enough.
- **Root README table is newest-first**: the top row is always the most
  recently added or updated recipe.
- **Only measured numbers.** Every figure in a README or table must come
  from a live run with the published flags, benchmarked with
  `tools/llm-bench` (fresh salt, warmup discarded). No projected or
  borrowed numbers.
- **Upstream-trail entries carry their date** (the PR/comment creation
  date, `YYYY-MM-DD`, at the start of the line), newest first.
- **No prebuilt images.** Dockerfiles only, pinned base digests, patch
  files published in-repo. Every patch names its upstream thread and its
  retirement condition.

## Writing style: avoid the AI tells (user, 2026-08-28)

Applies to EVERYTHING written for the user or the outside world (docs, PR bodies, issue reports, artifacts, commit messages, recipe guides, chat replies). These are the patterns to avoid:

- **HARD RULE: no em dashes.** Not for asides, not for emphasis, not before a conclusion. Use a comma, a period, parentheses, or a colon instead. This is the single most reliable AI tell and the user has banned it outright.

- **Verbs**: delve, underscore, leverage, utilize, navigate, foster, harness, unlock, unpack, streamline, elevate, empower, showcase, illuminate, bolster, spearhead, embark, champion, resonate, garner.
- **Adjectives**: robust, pivotal, crucial, nuanced, multifaceted, comprehensive, cutting-edge, transformative, revolutionary, seamless, dynamic, holistic, innovative, intricate, meticulous, vibrant, invaluable, profound, groundbreaking, paramount, ever-evolving, fast-paced, game-changing, unparalleled.
- **Nouns & metaphors**: tapestry, testament, realm, landscape, journey, beacon, cornerstone, synergy, paradigm, ecosystem, framework, scaffolding, plethora, myriad, symphony, catalyst, "through the lens of", north star, double-edged sword.
- **Claude-era tells**: load-bearing, scaffolding, broader, texture, "the shape of", "the core of", "the thing is", "does a lot of work", "quietly", "worth noting", "the honest answer", "genuinely", "frankly", "to be fair", "this matters because".
- **Sentence structures**: "It's not just X — it's Y"; "Here's the thing:" / "Here's why that matters:"; "The stakes couldn't be higher"; "Something real is happening"; "In many ways" / "At some level" / "Arguably"; "That said,"; "Let's break this down" / "Let's dive in"; rule-of-three triads ("clear, concise, and compelling"); rhetorical question followed by its immediate answer.
- **Punctuation & typography**: em dashes everywhere (several per paragraph, unspaced, where a comma/period/parens belongs); colons before every explanation; heavy semicolons; curly quotes/apostrophes where a human would type straight ones; dramatic ellipses…; empty parenthetical asides; rhetorical question marks; bold on random mid-sentence phrases; headers/bullets for what should be plain prose; emoji bullet markers (🚀 ✅ 💡).
- **Openers & closers**: "In today's ever-evolving/fast-paced world"; "In the realm of…"; "Certainly!" / "Great question!" / "Absolutely!"; "It's important to note that…" / "It's worth noting…"; "In summary / In conclusion / In essence / Ultimately"; "I hope this helps!" / "Let me know if you'd like…"; "Remember, …" as a moralizing wrap-up.
- **Transitions**: moreover, furthermore, additionally, consequently, notably, importantly, crucially, in light of this, with that in mind.
- **Hedge-and-balance filler**: "a nuanced approach", "strike a balance", "navigate the complexities", "a delicate balance", "there's no one-size-fits-all", "it depends on the context", "both sides have merit", "at the end of the day".
- **Tone tells**: every paragraph the same length and rhythm; tidy risk-free conclusions; zero concrete numbers, names, or examples; over-explaining the obvious.

The counter-pattern this rig already lives by: concrete numbers, real file paths, measured results, plain verbs, and conclusions that commit.
