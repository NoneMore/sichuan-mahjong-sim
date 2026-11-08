//! Settlement bookkeeping for the authoritative [`Game`].
//!
//! Payments, winner records, and terminal wall-exhaustion settlement remain
//! methods on the same game state. Only the private implementation lives here;
//! game progression and policy-visible decisions stay in the parent module.

use super::*;

impl Game {
    pub(super) fn pay_kong_opponents(&mut self, declarer: Seat, points: u32, kind: MeldKind) {
        for payer in Seat::ALL {
            if payer == declarer || self.winners[payer.index()] {
                continue;
            }
            self.transfer(payer, declarer, points);
            self.kong_ledger.push(KongRecord {
                declarer,
                payer,
                kind,
                points,
            });
        }
    }

    pub(super) fn transfer(&mut self, from: Seat, to: Seat, points: u32) {
        let points = points as i32;
        self.scores[from.index()] -= points;
        self.scores[to.index()] += points;
    }

    pub(super) fn settle_self_draw_win(&mut self, winner: Seat, kind: ActionKind) {
        let index = winner.index();
        let method = match kind {
            ActionKind::AfterDraw(DrawKind::Replacement) => WinMethod::AfterKong,
            _ => WinMethod::SelfDraw,
        };
        let hand = self.hands[index];
        let fan = rules::win_fan(&hand, &self.melds[index], method)
            .expect("legal Hu action has a winning shape");
        for payer in Seat::ALL {
            if payer != winner && !self.winners[payer.index()] {
                self.transfer(payer, winner, fan.total as u32);
            }
        }
        self.winners[index] = true;
        self.win_info[index] = Some(WinInfo {
            method,
            fan: fan.total,
            winning_tile: self.last_draw_tile,
            payer: None,
        });
    }

    pub(super) fn settle_discard_win(&mut self, winner: Seat, tile: Tile, method: WinMethod, payer: Seat) {
        let index = winner.index();
        let mut hand = self.hands[index];
        hand.add(tile)
            .expect("claimed Hu tile fits the winner's hand");
        let fan = rules::win_fan(&hand, &self.melds[index], method)
            .expect("claimable Hu tile produces a winning shape");
        self.transfer(payer, winner, fan.total as u32);
        self.winners[index] = true;
        self.win_info[index] = Some(WinInfo {
            method,
            fan: fan.total,
            winning_tile: Some(tile),
            payer: Some(payer),
        });
    }


    pub(super) fn end_hand(&mut self, reason: EndReason) {
        if self.result.is_some() {
            return;
        }
        if reason == EndReason::WallExhausted {
            self.settle_wall_exhaustion();
        }
        self.stage = Stage::Ended;
        self.pending = None;
        self.legal_actions.clear();
        self.result = Some(GameResult {
            scores: self.scores,
            winners: self.winners,
            end_reason: reason,
        });
    }

    fn settle_wall_exhaustion(&mut self) {
        let entries: [rules::WallExhaustionEntry; PLAYER_COUNT] = std::array::from_fn(|index| {
            let seat = Seat::new(index).expect("seat index");
            let is_hu = self.winners[index];
            let missing = self.missing[index].expect("missing suit declared before play");
            let is_flower_pig = !is_hu
                && rules::has_missing_suit_tiles(&self.hands[index], &self.melds[index], missing);
            let max_discard_fan = if !is_hu && !is_flower_pig {
                self.max_discard_fan(index, missing).unwrap_or(0)
            } else {
                0
            };
            rules::WallExhaustionEntry {
                seat,
                is_hu,
                is_flower_pig,
                is_tenpai: max_discard_fan > 0,
                max_discard_fan,
            }
        });

        for transfer in rules::settle_wall_exhaustion(&entries) {
            self.transfer(transfer.from, transfer.to, transfer.points);
        }

        let ledger = self.kong_ledger.clone();
        for seat in Seat::ALL {
            let entry = entries[seat.index()];
            if !rules::must_refund_kong_income(entry.is_hu, entry.is_flower_pig, entry.is_tenpai) {
                continue;
            }
            for record in &ledger {
                if record.declarer == seat {
                    let _ = record.kind;
                    self.transfer(seat, record.payer, record.points);
                }
            }
        }
    }

    fn max_discard_fan(&self, index: usize, missing: Suit) -> Option<u8> {
        let hand = self.hands[index];
        let melds = &self.melds[index];
        let mut best: Option<u8> = None;
        for tile in eval::waiting_tiles(&hand, melds) {
            if tile.suit() == missing {
                continue;
            }
            let mut winning = hand;
            if winning.add(tile).is_err() {
                continue;
            }
            if !rules::can_hu(&winning, melds, missing) {
                continue;
            }
            if let Some(fan) = rules::win_fan(&winning, melds, WinMethod::Discard) {
                best = Some(best.map_or(fan.total, |current| current.max(fan.total)));
            }
        }
        best
    }}
