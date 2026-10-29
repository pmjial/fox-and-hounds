# Ideas for further implementations

- **Book solver / strategy player.** Implement the "eyes closed" strategy from *Winning Ways* ch. 20 as a `BookPlayer`, or an exact memoised solver. Exhaustive search of the 8x8 game finishes in seconds and shows the hounds win from every fox start, even if the fox moves first.
- **Book conventions.** Let hounds move first and/or let the fox choose its starting square.
- **Stronger search.** Iterative deepening, transposition table, move ordering. Hound wins are many plies deep, so a fixed depth 4–8 horizon is the limiting factor.
- **Separate evaluation per side.** The fox and the hounds currently share one evaluation function.
