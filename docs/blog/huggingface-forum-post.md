# HuggingFace forum post — discussion framing

**Suggested category:** Show and Tell (fall back to Research / general if unavailable)

**Title:**

> Are failed agent traces actually usable as fine-tuning data? (a hypothesis + a prototype)

**Body:**

---

Hi all,

I've been sitting on a hypothesis and I'm not sure it holds up, so I'd rather put
it to people who actually fine-tune than keep guessing on my own.

**The hypothesis:** when a tool-using agent fails — picks the wrong tool, sends
malformed arguments, over-refuses a harmless request — that failure is a concrete
example of the behavior you'd want to train away. So in principle, the failures
piling up in your LangSmith/Langfuse logs are fine-tuning data you're throwing out.

That *sounds* right to me. But I genuinely don't know if it survives contact with
reality, and there are two things I'm unsure about.

**Question 1: are these failures actually usable as training data at all?**

Or does it backfire? My worry is that "corrected" failure traces might be lower
quality than they look — you're teaching the model a fix you inferred, not a fix
you verified. Has anyone tried training on agent failures? Did it help, do
nothing, or hurt? I'd love real experience here, positive or negative.

**Question 2: where should the line be for "correctable from the trace"?**

This is the part I keep going back and forth on. Some failures have an obvious
right answer *in the trace itself*: the agent called `calculator` for a weather
question when it had a `get_weather` tool — the fix is right there. But others
need external ground truth: "did the code actually pass its tests?", "is this fact
accurate?", "did the task complete?" — the trace alone can't tell you.

My instinct is: only generate a training pair when the fix is derivable from the
trace, and *skip* the rest rather than invent a plausible-looking answer, because
confidently-wrong labels seem worse than fewer labels. But I'm not confident that
boundary is in the right place — maybe it's too conservative and throws away
usable signal, or maybe even the "obvious" cases are shakier than I think.

**The prototype (context, not a pitch)**

To actually test this instead of just theorizing, I built a small local CLI —
[trace2train](https://github.com/wane528/trace2train) (`pip install trace2train`,
MIT). It detects the failures with rules, then does exactly the "skip if not
derivable" thing above, and outputs LLaMA-Factory SFT/DPO JSONL. It's very much
v0.1 — I've only run it on a small public-dataset slice and a synthetic Langfuse
run, so I can't claim it works, only that it lets me try the idea.

If you want to see the shape of it in 30s (no data or key needed):

```
pip install trace2train
trace2train inspect --demo
```

But honestly the tool is secondary. What I'm really after is whether the two
questions above have known answers I'm just not aware of. If this is a solved
problem, or a known-bad idea, I'd rather hear that now.

Thanks — genuinely curious what people think.
