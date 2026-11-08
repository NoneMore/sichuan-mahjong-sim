//! Diagnostic accessors and invariant checks for [`Game`].

use super::*;

impl Game {
    /// Diagnostic RNG state. Policies must not use it.
    pub fn debug_rng_state(&self) -> u64 {
        self.rng.state
    }

    /// Diagnostic tile conservation over the wall, hands, melds, and discards.
    ///
    /// This is an aggregate debug aid. Policies must not use it.
    pub fn debug_tile_counts(&self) -> [u8; TILE_KINDS] {
        let mut counts = [0u8; TILE_KINDS];
        for index in self.wall_head..self.wall_tail {
            counts[self.wall[index] as usize] += 1;
        }
        for seat in Seat::ALL {
            for tile in self.hands[seat.index()].tiles() {
                counts[tile.index() as usize] += 1;
            }
            for meld in &self.melds[seat.index()] {
                counts[meld.tile().index() as usize] += meld.tile_count();
            }
        }
        for discard in &self.discards {
            if discard.meld_claim.is_none() {
                counts[discard.tile.index() as usize] += 1;
            }
        }
        counts
    }

    /// Debug-only invariant check.
    ///
    /// Verifies that every tile kind occurs exactly four times across the
    /// documented ownership representation, that scores stay zero-sum, that
    /// winner metadata matches the winner flags, that every winner and every
    /// discard Hu marker reference each other consistently, and that
    /// discard/meld claims are consistent. Not a policy input, and not on any
    /// hot path: it runs only when a diagnostic caller asks for it.
    pub fn debug_validate(&self) -> Result<(), String> {
        if self.wall_head > self.wall_tail || self.wall_tail > TOTAL_TILES {
            return Err("wall bounds are inconsistent".to_string());
        }
        for (index, count) in self.debug_tile_counts().into_iter().enumerate() {
            if count != COPIES_PER_TILE {
                let tile = Tile::from_index(index as u8).expect("tile kind");
                return Err(format!(
                    "tile {tile} occurs {count} times instead of {COPIES_PER_TILE}"
                ));
            }
        }
        let score_sum: i32 = self.scores.iter().sum();
        if score_sum != 0 {
            return Err(format!("scores are not zero-sum: {score_sum}"));
        }
        for seat in Seat::ALL {
            let index = seat.index();
            if self.winners[index] != self.win_info[index].is_some() {
                return Err(format!("winner metadata mismatch for {seat}"));
            }
            let Some(info) = self.win_info[index] else {
                continue;
            };
            if info.fan == 0 {
                return Err(format!("winner {seat} has zero fan"));
            }
            // Every non-self-draw win must be recorded on exactly one discard
            // record and agree with that record's seat, tile, and robbery flag.
            let references: Vec<usize> = self
                .discards
                .iter()
                .enumerate()
                .filter(|(_, discard)| discard.winners[index])
                .map(|(position, _)| position)
                .collect();
            if info.method.is_self_draw() {
                if info.payer.is_some() {
                    return Err(format!("self-draw winner {seat} records a payer"));
                }
                if let Some(position) = references.first() {
                    return Err(format!(
                        "self-draw winner {seat} is recorded on discard {position}"
                    ));
                }
                continue;
            }
            let Some(payer) = info.payer else {
                return Err(format!("{seat} win by {:?} records no payer", info.method));
            };
            if references.len() != 1 {
                return Err(format!(
                    "{seat} win by {:?} is recorded on {} discard records instead of one",
                    info.method,
                    references.len()
                ));
            }
            let discard = &self.discards[references[0]];
            if discard.seat != payer {
                return Err(format!(
                    "{seat} win by {:?} is paid by the wrong discarder",
                    info.method
                ));
            }
            if info.winning_tile != Some(discard.tile) {
                return Err(format!(
                    "{seat} win by {:?} records a different winning tile",
                    info.method
                ));
            }
            if discard.robbed_kong != (info.method == WinMethod::RobbingKong) {
                return Err(format!(
                    "{seat} win by {:?} has a robbed-kong mismatch",
                    info.method
                ));
            }
        }
        for (position, discard) in self.discards.iter().enumerate() {
            if discard.meld_claim.is_some() && discard.winners.iter().any(|&winner| winner) {
                return Err("discard is both meld-claimed and Hu-claimed".to_string());
            }
            if let Some(claimant) = discard.meld_claim {
                let found = self.melds[claimant.index()].iter().any(|meld| {
                    meld.tile() == discard.tile
                        && meld.from() == Some(discard.seat)
                        && matches!(
                            meld.kind(),
                            MeldKind::Peng | MeldKind::OpenKong | MeldKind::AddedKong
                        )
                });
                if !found {
                    return Err("discard meld claim has no matching meld".to_string());
                }
            }
            // The reverse direction: every discard Hu marker needs a global
            // winner whose metadata points back at this record.
            for seat in Seat::ALL {
                let index = seat.index();
                if !discard.winners[index] {
                    continue;
                }
                let Some(info) = self.win_info[index] else {
                    return Err(format!(
                        "discard {position} marks {seat} as a winner without winner metadata"
                    ));
                };
                if info.method.is_self_draw() {
                    return Err(format!(
                        "discard {position} marks {seat} as a {:?} winner",
                        info.method
                    ));
                }
                if info.payer != Some(discard.seat) {
                    return Err(format!(
                        "discard {position} winner {seat} is paid by the wrong seat"
                    ));
                }
                if info.winning_tile != Some(discard.tile) {
                    return Err(format!(
                        "discard {position} winner {seat} records a different winning tile"
                    ));
                }
                if discard.robbed_kong != (info.method == WinMethod::RobbingKong) {
                    return Err(format!(
                        "discard {position} winner {seat} has a robbed-kong mismatch"
                    ));
                }
            }
        }
        Ok(())
    }
}
