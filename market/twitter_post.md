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

Regent is not a transformer.

Every frontier model today is built on the same architecture from a 2017 paper. Regent is a Mamba-2 state-space model with GQA attention at selected layers, a verification head, a structured knowledge graph interface, and a persistent memory system.

Different engine. Different properties. Different outcomes.

---

**Tweet 5**

The second problem: AI is priced as a service.

Per token. On someone else's servers. Over the internet. In USD.

For a hospital in Lagos or a legal clinic in Dhaka, that is not a viable economic model. It is a permanent dependency on infrastructure they do not control.

---

**Tweet 6**

Regent ships as weights and code.

License once. Deploy on your own hardware. Run indefinitely.

No per-inference fees. No internet required. No data leaving your infrastructure.

The 7B model runs on a $500 edge device.

---

**Tweet 7 (benchmark table)**

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

**Tweet 8**

Two tiers:

Regent (7B to 50B) is open source. Free to deploy. Built for organizations that cannot absorb ongoing API costs.

Grande Regent (70B to 1T) is commercial. Frontier scale. Enterprise tooling and support. Through Alchymia Groom.

Same architecture. Workflows built on one migrate to the other.

---

**Tweet 9**

The markets:

Legal. Healthcare. Defense and government. Robotics. Drones. Pharma. Nuclear. Maritime. Mining. Insurance. Compliance. Emergency services. Audit. Agriculture. Industrial automation.

What they share: the cost of a wrong answer is measurable, and cloud dependency is either too expensive or not allowed.

---

**Tweet 10**

Regent is the first real language model to come out of Africa.

Not a wrapper. Not a fine-tune. A ground-up architecture, built to be on par with the best models in the world at the workloads it is designed for.

Alchymia AI Research Labs is founded by Ayomide I. Daniels. The team is in the diaspora. The work is global.

https://alchymia.ai

---

**Tweet 11**

We are also building DSTP, a Distributed Shared Training Protocol.

It is designed to enable training models at 1 to 2 trillion parameters by pooling compute across institutions and geographies.

The goal: frontier-scale models should not require a $200M check to a single cloud provider.

More on this soon.

---

**Tweet 12**

The core belief at Alchymia: developing economies do not need cheaper versions of Western AI. They need AI with different properties.

Offline. Owned. Auditable. Affordable.

This is not charity. It is the 10x magnitude of ingenuity it takes to compete from where we stand. That is the ethos.

---

**Tweet 13 (CTA)**

Regent is available now on HuggingFace.

Weights, code, architecture docs, and training pipeline are all open.

[link to repo]

https://alchymia.ai
research@alchymia.ai for enterprise and Grande Regent.

---
