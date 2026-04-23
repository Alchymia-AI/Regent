# X / Twitter Release Post

*Format: launch thread. Each block is one tweet.*

---

**Tweet 1 (anchor)**

We just open-sourced Regent.

The first production language model built in Africa. Built for decisions that matter.

It thinks before it speaks. It calls tools. It tells you which words to trust. And it runs on your hardware, offline, forever.

Thread on what it does and why it exists.

---

**Tweet 2**

AI models today generate text and hope you check it.

In a hospital, that hope costs a life. In a courtroom, it costs a case. On a ship, it costs a hull.

Regent scores the accuracy of every word as it writes. When confidence drops, it stops, retrieves what it needs, and tries again. One pass. No second model.

---

**Tweet 3**

Regent also thinks before it answers.

When the question is complex, it reasons internally first, then responds. You see the reasoning. You see the answer. You see the confidence score on every word of both.

It calls external tools natively. APIs, databases, search. The model decides when it needs outside information, calls for it, and folds the result back in.

These are not plugins. They are architecture.

---

**Tweet 4**

Your knowledge base updates. The model sees it immediately. No retraining. No re-deployment.

New drug interaction? Reflected on the next query. Crop disease outbreak? Updated before the next field decision. New fraud pattern? Available immediately. Sanctions list change? In the next AML check. Commodity price shift? In the next risk model. New case law? In the next brief. Equipment fault code? In the next maintenance decision. Patient history update? In the next clinical recommendation. Route hazard? In the next navigation plan.

Every industry where decisions depend on current information. No retraining. No delay.

For most use cases, this replaces RAG entirely. No vector database. No embedding pipeline. No chunking. No retrieval misses. No context window pressure. Knowledge goes in structured and scored. The model reads it natively.

Other models bake knowledge into weights during training. Updating them means months and millions of dollars. Regent reads your live knowledge graph as native input, every request, in real time.

---

**Tweet 5**

Regent is not a transformer.

Every frontier model today is built on the same architecture from a 2017 paper. Regent is a Mamba-2 state-space model with GQA attention at selected layers, a verification head, a structured knowledge graph interface, and a persistent memory system.

Different engine. Different properties. Different outcomes.

---

**Tweet 6**

Every other model runs every layer on every token. Whether the token needs deep reasoning or is just continuing a sentence. Same cost. Same compute. Every time.

Regent has an adaptive gate. The model learns which tokens need its expensive attention layers and which ones don't. When attention isn't needed, it skips it. When it is, it fires.

The result: lower cost per token in production, faster inference on routine text, and full reasoning power exactly when the task demands it. The model allocates its own compute.

---

**Tweet 8**

The second problem: AI is priced as a service.

Per token. On someone else's servers. Over the internet. In USD.

For a hospital in Lagos or a legal clinic in Dhaka, that is not a viable economic model. It is a permanent dependency on infrastructure they do not control.

---

**Tweet 7**

Regent ships as weights and code.

License once. Deploy on your own hardware. Run indefinitely.

No per-inference fees. No internet required. No data leaving your infrastructure.

The 7B model runs on a single server. No cluster required.

---

**Tweet 9 (benchmark table)**

```
Capability comparison

                          GPT-5  Claude    Llama 3.1  Mistral  Regent 7B
                                Opus 4.6    70B       Large
Real-time accuracy score   No     No         No        No        Yes
Thinks before answering    No    Yes         No        No        Yes
Native tool calling        Yes   Yes        Yes       Yes        Yes
Stops when uncertain       No     No         No        No        Yes
Runs fully offline         No     No        Yes       No*        Yes
Fixed memory any session   No     No         No        No        Yes
Native graph/KB input      No     No         No        No        Yes
Per-token cost after dep   Yes    Yes        No        Yes        No
Air-gap compatible         No     No        Yes        No        Yes
Non-transformer arch       No     No         No        No        Yes
Open source                No     No        Yes        No        Yes
```

*Mistral has self-hosted options but the frontier flagship is API-only.

GPT-5 and Opus 4.6 are more capable on general chat. That is not the comparison. The question is whether they can do all of these things together, natively, on your hardware. They cannot.

---

**Tweet 10**

Two tiers:

Regent (7B to 50B) is open source. Free to deploy. Built for organizations that cannot absorb ongoing API costs.

Grande Regent (70B to 1T) is commercial. Frontier scale. Enterprise tooling and support. Through Alchymia Groom.

Same architecture. Workflows built on one migrate to the other.

---

**Tweet 11**

Code generation, but the model holds your entire repo in context. Not 128K tokens. The whole thing.

It calls your compiler mid-generation. Runs your tests. Gets the failures back. Fixes the code. One pass.

The verification head tells you which lines it's confident about and which ones it's guessing. Before you run anything.

No other code model does all three natively.

---

**Tweet 13**

The markets:

Legal. Healthcare. Defense and government. Robotics. Drones. Pharma. Nuclear. Maritime. Mining. Insurance. Compliance. Emergency services. Audit. Agriculture. Industrial automation. Code generation.

What they share: the cost of a wrong answer is measurable, and cloud dependency is either too expensive or not allowed.

---

**Tweet 12**

Regent is the first real language model to come out of Africa.

Not a wrapper. Not a fine-tune. A ground-up architecture, built to be on par with the best models in the world at the workloads it is designed for.

Alchymia AI Research Labs is founded by Ayomide I. Daniels. The team is in the diaspora. The work is global.

https://alchymia.ai

---

**Tweet 13**

We are also building DSTP, a Distributed Shared Training Protocol.

It is designed to enable training models at 1 to 2 trillion parameters by pooling compute across institutions and geographies.

The goal: frontier-scale models should not require a $200M check to a single cloud provider.

More on this soon.

---

**Tweet 16**

The core belief at Alchymia: developing economies do not need cheaper versions of Western AI. They need AI with different properties.

Offline. Owned. Auditable. Affordable.

This is not charity. It is the 10x magnitude of ingenuity it takes to compete from where we stand. That is the ethos.

---

**Tweet 17**

Coming in the next few months: Darkhorse.

Alchymia's Large Language Action Model (LLAM). First generation. Our flagship generalist AI.

Millions of tokens of context. The same fixed memory efficiency as Regent. Built to act, not just answer. The model that does the work, not the model that drafts the memo about the work.

Today, an "AI agent" is a prompt chain duct-taped to an LLM that was never designed to act. Darkhorse is a model built from the ground up to decide and execute. Not a chat model with agent scaffolding on top. An action model, out of the box.

Millions of tokens of context. Compact memory efficiency. Makes decisions and carries them out.

More soon.

---

**Tweet 18 (CTA)**

Regent is available now on HuggingFace.

Weights, code, architecture docs, and training pipeline are all open.

[link to repo]

https://alchymia.ai
research@alchymia.ai for enterprise and Grande Regent.

---
