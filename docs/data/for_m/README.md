# For M: what the agents say (dictionary + one episode), exported 2026-11-09

Free (frozen checkpoints, laptop). Seven folders, each with `dictionary.csv/json` and `transcript.csv/json`:

| folder | checkpoint | what it is |
|---|---|---|
| `REALNET_PS51_final` | real network, confirmation seed 51, update 450 | **the strongest:** passed the real-net confirmation (redirection x18.7) |
| `S2VQ_s193_u2000` | toy, step-2 VQ seed 193, update 2,000 | the cleanest toy code |
| `C1T3_s180_u2000` | toy, C1-T3 seed 180, update 2,000 (before its listener reset) | wire-blind toy code under turnover |
| `ORACLE0_start` | toy starting point of every toy run (before any communication training) | **contrast: no code yet** |

**The world.** A sender who can see which of 4 spots ("fires") is lit, and a listener who can't. Each step the
sender broadcasts 3 tokens (64 possible tokens per slot); the listener hears them one step later. A Craft next to
the lit spot delivers and pays both.

**Dictionary** (per lit spot): the sender's most common 3-token message and how often it uses it; each slot's most
common token ("slot word") and its share; `read_back_correct` = how often a simple reader (per-slot frequencies)
recovers the lit spot from one message.
- **Real net seed 51:** one word per spot, used 81-99.6% of the time; read back 99.6-100%.
- **Toy (S2VQ s193):** spots 1 and 3 have one word at 100%, spot 2 at 67% (one slot varies). **Spot 0 has no word of
  its own:** 767 different messages, none of them the other spots' words, and it still reads back 100%. It means
  "not 1, 2 or 3".
- **ORACLE0 start:** nothing (most common message 0.1%; read back 23-36%, chance 25%). The words were learned.
- The toy words for spots 1-3 (63-1-24, 28-50-40, 49-10-3) recur across independently trained seeds, because every
  toy run starts from the same ORACLE0 weights and gets the same training push. The real net's words are different.

**Transcript** (one world, chosen as the one with the most listener deliveries in a fresh rollout): per step, the
lit spot and its position, where the sender was, the triple it sent and what it reads as, where the listener was,
what it heard (the sender's previous triple), what it did, how many moves it was from the lit spot, and deliveries.
128 steps (the real net's rollout is 512 steps long; the first 128 are exported). In the toy, the listener often
waits near a spot and Crafts only after hearing that spot's word (e.g. S2VQ steps 14-16, 34-36). "idle (k)" = an
action with no movement in this world.

## Added Nov 9: the C1-E replication's surviving codes at update 3,000 (`C1REP_s231/s232/s234_u3000`)

These are the three toy seeds that still have a code the listener relies on, 1,500 updates after the training
nudge was removed (Cam's verdict: a self-sustaining code, met but fragile).
- **Diffuse:** each spot's most common message is at most 1.4% of its messages, and a slot's word for a spot covers
  8-23%. A simple reader recovers the spot from one message 4-95% of the time, depending on spot and seed.
- **Their own words:** the shared starting-point words (63-1-24 etc.) are gone. Each seed now uses different
  tokens.
- **Still read:** swapping in another spot's message costs listeners 49-80% of deliveries, while swapping in
  another message for the same spot costs nothing.

**Caveats.**
- Seed 51 predates the wire-blind fix: its sender's words depend on the fire *and* on what the partner just said
  (protocol, step A). In normal play that combination is stable, so each spot gets one word.
- The 3 slots are redundant copies of one meaning, not a sentence.
- Descriptive, not a registered result.
