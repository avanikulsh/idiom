# Qualitative Analysis: Cross-Lingual Idiom Matching Quality

## Summary of Match Types

Based on manual inspection of top-scoring matches, we identified **8 distinct categories** of match quality:

---

## ✅ 1. Perfect Equivalents (True Positives)

**What works:** Shared proverbs or conceptually identical metaphors across languages.

### Examples:

**English ↔ Japanese:**
- "easier said than done" ↔ "言うは易く行うは難し" (0.7515)
  - **Verdict:** ✔️ Perfect - Same proverb structure and meaning

**English ↔ French:**
- "go bananas" ↔ "avoir la banane" (0.7075)
  - **Verdict:** ✔️ Excellent - Both use fruit metaphor for excitement

**English ↔ Finnish:**
- "like peas in a pod" ↔ "kuin kaksi marjaa" (0.6444)
  - **Verdict:** ✔️ Good - Both compare similarity to paired foods

**Why this works:** Embeddings correctly captured semantic similarity when:
- Languages share cultural proverbs
- Metaphorical mappings align (emotion → fruit, similarity → food pairs)
- Usage contexts overlap significantly

---

## ⚠️ 2. Antonymous Metaphors (False Positives)

**What fails:** Embeddings conflate opposite emotions because they share domain/intensity.

### Examples:

**English ↔ Japanese:**
- "make your blood boil" (anger) ↔ "血が引く" (your blood runs cold - fear/shock) (0.7524)
  - **Verdict:** ❌ Wrong - Opposite emotional valence
  - **Why:** Model sees "blood + emotion intensity" = similar

**English ↔ French:**
- "have a cow" (don't overreact) ↔ "avoir la chair de poule" (goosebumps - fear) (0.7606)
  - **Verdict:** ❌ Wrong - Different emotion class (annoyance vs fear)

**Why this fails:**
- Embeddings capture domain similarity (emotion, body metaphor)
- But lack nuance to distinguish antonyms or emotion polarity
- High cosine similarity despite opposite meanings

---

## ❌ 3. Lexical Overlap Without Semantic Equivalence

**What fails:** Model matches on shared words/body parts, not meaning.

### Examples:

**English ↔ Japanese (all ~0.70-0.78 similarity):**
- "in one ear and out the other" (forgetfulness) matched to:
  - "耳に挟む" (overhear)
  - "耳を塞ぐ" (cover your ears)
  - "耳を澄ます" (listen carefully)
  - "耳が遠い" (hard of hearing)

  **Verdict:** ❌ All wrong - Just lexical matching on "ear" (耳)

**English ↔ Japanese:**
- "put your heads together" ↔ "頭が固い" (stubborn-headed) (0.7574)
  - **Verdict:** ❌ Wrong - Shared "head" but opposite concepts (collaboration vs stubbornness)

**Why this fails:**
- Embeddings heavily weight lexical overlap
- Idioms with same body part clustered together
- Metaphorical meaning ignored in favor of surface tokens

---

## ⚠️ 4. Sentiment Match, Metaphor Mismatch

**What's partial:** Emotion/concept correct, but metaphorical imagery differs.

### Examples:

**English ↔ Japanese:**
- "bite someone's head off" (snap angrily) ↔ "頭に来る" (anger rises to head) (0.7395)
  - **Verdict:** ⚠️ Semantic match (anger) but different metaphors (violence vs location)

**English ↔ Japanese:**
- "do someone's head in" (annoy) ↔ "尻を叩く" (kick butt - motivate) (0.7943)
  - **Verdict:** ⚠️ Both intense/forceful but different target emotions

**Why this is partial:**
- Good for cross-lingual sentiment analysis
- Not suitable for idiom-to-idiom translation
- Useful for paraphrase but not equivalence

---

## ❌ 5. Non-Idiomatic Literal Actions

**What fails:** Idiomatic expression matched to literal action phrase.

### Examples:

**English ↔ Japanese:**
- "stick your neck out" (take a risk) matched to:
  - "声を立てる" (make a sound) (0.7251)
  - "首を振る" (shake your head) (literal body movement)

  **Verdict:** ❌ Wrong - Risk-taking vs physical gestures

**English ↔ Finnish:**
- "shake a leg" (hurry up) ↔ "搖頭晃腦" (shake head and brain - literally) (0.6672)
  - **Verdict:** ❌ Wrong - Idiomatic urgency vs literal movement

**Why this fails:**
- One language's idiom = other language's literal description
- Embeddings can't distinguish figurative vs compositional meaning
- Body part + action = spurious similarity

---

## ⚠️ 6. Contextual Match, Not Idiom Match

**What's misleading:** Usage contexts semantically similar, but idioms aren't equivalent.

### Examples:

**English ↔ Japanese:**
- "over your head" (too complex) ↔ "気を使うな" (don't mind me)
  - **Similarity based on:** Conversational contexts where someone's being dismissive/talking past someone
  - **Verdict:** ⚠️ Context overlap ≠ idiom equivalence

**Why this happens:**
- Our representation: `idiom + contexts` (3 sentences)
- Model weights context heavily
- Good for: Document similarity, scene understanding
- Bad for: Idiom dictionary lookup

**Implication:** Confirms our symmetric design works for usage-based semantics, but may overfit to context genre (movie subtitles vs formal BNC).

---

## ⚠️ 7. Partial Analogies / Action Overlap

**What's partial:** Physical action or scenario similar, but idiomatic meaning diverges.

### Examples:

**English ↔ French:**
- "jeter l'éponge" (throw in the towel - give up) ↔ "chuck it down" (throw in bin)
  - **Verdict:** ⚠️ Both involve throwing/discarding, but different idiom classes

**English ↔ French:**
- "say cheese" (smile for photo) ↔ "en faire tout un fromage" (make a big cheese out of it - exaggerate) (0.6957)
  - **Verdict:** ⚠️ Both mention cheese, but unrelated meanings

**Why this happens:**
- Embeddings cluster by action scripts (throwing, food mentions)
- Metaphorical target differs despite surface similarity

---

## ❌ 8. Complete Mismatches (Embedding Artifacts)

**What fails:** No semantic, metaphorical, or lexical justification.

### Examples:

**English ↔ Japanese:**
- "meet your maker" (die) ↔ "一手" (one hand/move - from games)
  - **Verdict:** ❌ Completely unrelated

**English ↔ Finnish:**
- "right as rain" ↔ "sataa kuin Esterin perse" (raining like Esther's ass)
  - **Verdict:** ⚠️ Both weather-related but semantically distant

**Why this happens:**
- Statistical noise in high-dimensional space
- Low-frequency idioms with sparse contexts
- Genre/domain effects (formal BNC vs casual subtitles)

---

## Quantitative Breakdown (Manual Annotation Needed)

To properly evaluate, we would need:

1. **Gold standard annotations:** Bilingual speakers mark true equivalents
2. **Precision@K calculation:** What % of top-K matches are valid?
3. **Category distribution:** How many fall into each of the 8 categories?

### Estimated Distribution (from manual inspection of top 30 matches):

| Category | Count (est.) | % |
|----------|--------------|---|
| ✅ Perfect equivalents | 4-6 | ~15-20% |
| ⚠️ Sentiment match only | 6-8 | ~20-25% |
| ❌ Lexical overlap | 5-7 | ~15-20% |
| ❌ Antonymous metaphor | 2-3 | ~5-10% |
| ⚠️ Contextual match | 4-6 | ~15-20% |
| ⚠️ Partial analogy | 2-4 | ~5-10% |
| ❌ Literal vs idiom | 2-3 | ~5-10% |
| ❌ Complete mismatch | 1-2 | ~5% |

**Takeaway:** Only ~15-20% are high-quality idiom equivalents. Remaining ~80% are semantically related but not substitutable for translation.

---

## Implications for Research

### What We Learned:

1. **Embeddings ARE capturing cross-lingual semantics**
   - Even "wrong" matches show systematic patterns (body parts, emotions, actions)
   - Not random noise

2. **Context-based representations conflate usage similarity with equivalence**
   - Good: Captures how idioms function in discourse
   - Bad: Can't distinguish metaphorical structure from situational overlap

3. **Lexical grounding is too strong**
   - Shared words (ear, head, blood) dominate similarity
   - Need metaphor-aware representations

4. **Language pairs differ in match quality**
   - Japanese > Finnish > French might reflect:
     - Dataset size (more chances for good matches)
     - Cultural distance (shared proverbs with English)
     - Context genre mismatch (BNC vs subtitles)

### Recommendations for Improvement:

1. **Metaphor-aware embeddings:** Pre-train on metaphor datasets (VUA, MOH-X)
2. **Idiom-specific encoders:** Separate idiom from context during encoding
3. **Contrastive learning:** Train with antonym pairs to distinguish opposition
4. **Cross-lingual grounding:** Use bilingual idiom dictionaries for supervision
5. **Evaluation with gold standard:** Create annotated test set for precision/recall

---

## Conclusion

Multilingual sentence transformers with context-based representations show **promising but noisy** cross-lingual idiom matching:

- ✅ **Strengths:** Captures semantic domains, usage patterns, sentiment
- ❌ **Weaknesses:** Lexical bias, metaphor conflation, antonym confusion
- 📊 **Estimated precision:** ~15-20% for true equivalents in top matches

This validates that embeddings capture *something* meaningful about cross-lingual idiom semantics, but are **not yet suitable** for building idiom translation dictionaries without human curation.
