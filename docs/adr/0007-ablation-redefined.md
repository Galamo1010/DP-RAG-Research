# ADR 0007 — What the third comparison measures

Date: 2026-08-26
Status: Accepted. Replaces the framing of the ablation in the proposal's 比較對象.

## Context

The proposal describes the third comparison twice, and the two descriptions do not
survive contact with the measurements in the same way.

**比較對象** frames it as a bound:

> 3. 策略A vs策略B（消融分析）：**以策略A觸發率作為嚴格上界**，量化策略B在
> 「argmax不同但top-k高度重疊」位置的**額外覆蓋增益**

This assumes strategy B fires wherever A fires and sometimes more, so that A caps
B and the interesting quantity is B's surplus. Both halves are false.

**Strategy B is not a superset of A.** Two distributions can share a top-k *set*
while ordering it differently, so B can fire where A does not and A can fire where
B does not. This is pinned by a test and was known before any GPU run.

**A does not bound B in either direction.** Measured at eps=10 over 184 queries:

| strategy | trigger rate |
|---|---|
| B (k=20, tau=0.9) | 14.4% |
| B (k=20, tau=0.7) | 59.1% |
| **A** | **82.5%** |
| B (k=50, tau=0.5) | 95.3% |

B spans 14.4% to 95.3% and A sits in the middle of that range. It is neither a
ceiling nor a floor; it is one configuration among many.

**Stage 3.2's own 核心任務 is already compatible** — it says "量化兩者在觸發率與
品質上的取捨", which asks for a trade-off rather than a bound. So what needs
replacing is the framing in 比較對象, not the comparison itself.

## Decision

The third comparison measures two things.

### 1. The trade-off surface

Strategy A and every Strategy B configuration are placed on the same
(epsilon saved x quality) plane. A is one point among them rather than a
reference line, and the Pareto frontier is read off the whole set — which is what
the proposal asked for in its next clause anyway.

This answers "which configuration is worth using", and it does so without needing
A to bound anything.

### 2. The overlap structure, position by position

For a given query, each position is classified by what both strategies would have
decided:

|  | B fires | B does not |
|---|---|---|
| **A fires** | ? | ? |
| **A does not** | ? | ? |

The two off-diagonal cells are the honest version of "額外覆蓋增益": positions B
reaches that A does not, **and** positions A reaches that B does not. The
proposal's framing had a name for the first and no name for the second, because it
assumed the second was empty. It is not.

This is what explains *why* the trade-off surface looks as it does — whether a
configuration wins by covering different positions or simply by being looser.

**This became possible only with `dprag/trace.py`.** A 2x2 over positions needs
each strategy's per-position decision on the same query, and until that module the
results held only per-answer rates. The measurement is not new because the
question changed; it is new because the record finally supports it.

## Consequences

**One falsified assumption to report, not hide.** The proposal predicted A would
cap B. It does not, and the reason is structural rather than incidental: top-k
Jaccard compares set membership while argmax compares an ordering, and neither
refines the other. Stating that is a finding.

**tau reads looser than it is, and that belongs beside the numbers.** At k=10,
tau=0.7 requires 9 of 10 tokens to match, which is why several B configurations
trigger far *below* A rather than above it. A reader who assumes tau=0.7 means "70%
similar" will misread the whole table.

**The 2x2 needs both strategies routed over the same query.** They walk different
trajectories once they disagree, so the classification is only exact at positions
before the first divergence, and approximate after it. Report the divergence point
alongside the table rather than presenting the cells as if both strategies had
seen the same context throughout.

**Nothing about the phase plan changes.** ADR 0003's three phases stand; this ADR
changes what is computed from their output, not what is run.
