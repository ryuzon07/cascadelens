The method misses tolerance on Sock Shop by 1.4 points (+0.064 against 0.05) and passes on the other
two systems. The stop rule applies: the gap is recorded, this configuration is kept, and the method
was not changed.

**Cause: the ranking barely depends on the case.** Our AC@1 is exactly 0.200 on Online Boutique and
Sock Shop. Each system has five injected services with equal case counts, so that is the score of
always naming one service. A check on 18 cases per system confirmed it: the top service was `carts`
in 18 of 18 Sock Shop cases and `cartservice` in 17 of 18 Online Boutique cases. The graphs are not
empty (about 300 edges), and the output is not plain column order.

**The published table shows the same behaviour.** In ASE 2024 Table 5, the Sock Shop 2 cells read
0.48, 0.46, 0.49, 0.45, 0.36 identically for PC-RW, Granger-RW, ICALiNGAM-RW, NTLR-RW and fGES-RW,
which use different causal discovery algorithms. A score that barely depends on the graph moves with
small changes in which service the walk favours, so a gap of a few points is unsurprising.

**Seeds would not help.** Changing the seed left the top ten unchanged in all 36 checked cases, so
one seed is used, not the several the plan expected.

**Cell gaps.** Online Boutique DELAY is 0.30 against 0.14, and Train Ticket CPU is 0.06 against
0.16. The cause was not established. Library drift since 2024 and the unstated data length are the
candidates in the plan; the changelog records no change to this method specifically.

**Reading this method.** It meets the selection criteria, but it is a weak comparator: its scores
reflect one fixed answer, not diagnosis. It says little about causal-graph methods in general.
