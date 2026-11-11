//! Settlement bookkeeping for the authoritative [`Game`].
//!
//! Payments, winner records, and wall-exhaustion refunds live here. Terminal
//! transitions and game progression stay in the parent module.

use super::*;

impl Game {
    pub(super) fn pay_kong_opponents(&mut self, declarer: Seat, kind: MeldKind) {
        let payment = rules::kong_payment(kind, self.active_count() - 1);
        for payer in Seat::ALL {
            if payer == declarer || self.winners[payer.index()] {
                continue;
            }
            self.transfer(payer, declarer, payment.per_payer);
            self.kong_ledger.push(KongRecord {
                declarer,
                payer,
                kind,
                points: payment.per_payer,
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
        let payment = rules::win_payment(fan.total, method, self.active_count() - 1);
        for payer in Seat::ALL {
            if payer != winner && !self.winners[payer.index()] {
                self.transfer(payer, winner, payment.per_payer);
            }
        }
        self.winners[index] = true;
        self.win_info[index] = Some(WinInfo {
            method,
            fan: payment.fan,
            winning_tile: self.last_draw_tile,
            payer: None,
        });
    }

    pub(super) fn settle_discard_win(
        &mut self,
        winner: Seat,
        tile: Tile,
        method: WinMethod,
        payer: Seat,
    ) {
        let index = winner.index();
        let mut hand = self.hands[index];
        hand.add(tile)
            .expect("claimed Hu tile fits the winner's hand");
        let fan = rules::win_fan(&hand, &self.melds[index], method)
            .expect("claimable Hu tile produces a winning shape");
        let payment = rules::win_payment(fan.total, method, self.active_count() - 1);
        self.transfer(payer, winner, payment.per_payer);
        self.winners[index] = true;
        self.win_info[index] = Some(WinInfo {
            method,
            fan: payment.fan,
            winning_tile: Some(tile),
            payer: Some(payer),
        });
    }

    pub(super) fn settle_wall_exhaustion(&mut self) {
        let entries: [rules::WallExhaustionEntry; PLAYER_COUNT] = std::array::from_fn(|index| {
            let seat = Seat::new(index).expect("seat index");
            let is_hu = self.winners[index];
            let missing = self.missing[index].expect("missing suit declared before play");
            let is_flower_pig = !is_hu
                && rules::has_missing_suit_tiles(&self.hands[index], &self.melds[index], missing);
            let max_discard_fan = if !is_hu && !is_flower_pig {
                // Non-flower-pig hands and melds contain no missing-suit
                // tiles, so the shape-only query matches the legal discard
                // wins.
                rules::max_discard_win_fan(&self.hands[index], &self.melds[index]).unwrap_or(0)
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
}
