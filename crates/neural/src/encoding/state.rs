//! Player-visible state extraction from one pending decision.

use sichuan_mahjong_sim::{
    Decision, Meld, Observation, PlayerView, Seat, PLAYER_COUNT, TILE_KINDS,
};

use super::schema::{
    discard_category_id, exchange_direction_id, meld_kind_id, meld_source_id, missing_suit_id,
    scaled_counts, stage_id, ContextFeatures, MeldSlot, PlayerFeatures, ResponseFeatures,
    StateFeatures, DISCARD_CATEGORIES, EXCHANGE_DIRECTION_FEATURES, HAND_TOTAL_DIVISOR,
    MELD_KIND_FEATURES, MELD_SLOTS, MELD_SOURCE_FEATURES, MISSING_SUIT_FEATURES, SCORE_DIVISOR,
    STAGE_FEATURES, TILE_FEATURES, WALL_DIVISOR,
};

pub(super) fn encode_state(decision: &Decision<'_>) -> StateFeatures {
    let observation = &decision.observation;
    let decider = decision.seat;

    let mut discard_counts = [[[0u8; TILE_KINDS]; DISCARD_CATEGORIES]; PLAYER_COUNT];
    for record in observation.discards {
        let source = decider.clockwise_distance(record.seat) as usize;
        let category = discard_category_id(record);
        let slot = &mut discard_counts[source][category][record.tile.index() as usize];
        *slot = slot.saturating_add(1);
    }
    let discards: [[[f32; TILE_FEATURES]; DISCARD_CATEGORIES]; PLAYER_COUNT] =
        std::array::from_fn(|source| {
            std::array::from_fn(|category| scaled_counts(&discard_counts[source][category]))
        });

    StateFeatures {
        own_hand: scaled_counts(observation.own_hand.counts()),
        players: std::array::from_fn(|relative| {
            let seat = seat_at_relative(decider, relative);
            player_features(player_view(observation, seat), decider)
        }),
        discards,
        context: context_features(observation),
        response: response_features(observation, decider),
    }
}

fn player_features(view: &PlayerView<'_>, decider: Seat) -> PlayerFeatures {
    let mut missing_suit = [0.0f32; MISSING_SUIT_FEATURES];
    missing_suit[missing_suit_id(view.missing_suit)] = 1.0;

    PlayerFeatures {
        concealed_total: view.concealed_count as f32 / HAND_TOTAL_DIVISOR,
        missing_suit,
        score: view.score as f32 / SCORE_DIVISOR,
        has_won: flag(view.has_won),
        melds: meld_slots(view.melds, decider),
    }
}

fn meld_slots(melds: &[Meld], decider: Seat) -> [MeldSlot; MELD_SLOTS] {
    // A valid hand declares at most four melds; fabricated extra melds are
    // outside the supported contract and dropped here.
    let mut ordered: [Option<Meld>; MELD_SLOTS] = [None; MELD_SLOTS];
    for (slot, &meld) in ordered.iter_mut().zip(melds) {
        *slot = Some(meld);
    }
    ordered.sort_by_key(|slot| slot.map_or(u8::MAX, |meld| meld.tile().index()));

    std::array::from_fn(|index| match ordered[index] {
        Some(meld) => meld_slot(meld, decider),
        None => zero_meld_slot(),
    })
}

fn meld_slot(meld: Meld, decider: Seat) -> MeldSlot {
    let mut tile = [0.0f32; TILE_FEATURES];
    tile[meld.tile().index() as usize] = 1.0;

    let mut kind = [0.0f32; MELD_KIND_FEATURES];
    kind[meld_kind_id(meld.kind())] = 1.0;

    let mut source = [0.0f32; MELD_SOURCE_FEATURES];
    source[meld_source_id(meld.from(), decider)] = 1.0;

    MeldSlot {
        valid: 1.0,
        tile,
        kind,
        source,
    }
}

fn zero_meld_slot() -> MeldSlot {
    MeldSlot {
        valid: 0.0,
        tile: [0.0; TILE_FEATURES],
        kind: [0.0; MELD_KIND_FEATURES],
        source: [0.0; MELD_SOURCE_FEATURES],
    }
}

fn context_features(observation: &Observation<'_>) -> ContextFeatures {
    let mut exchange_direction = [0.0f32; EXCHANGE_DIRECTION_FEATURES];
    exchange_direction[exchange_direction_id(observation.exchange_direction)] = 1.0;

    let mut stage = [0.0f32; STAGE_FEATURES];
    stage[stage_id(observation.stage)] = 1.0;

    ContextFeatures {
        wall_count: observation.wall_count as f32 / WALL_DIVISOR,
        exchange_direction,
        stage,
    }
}

fn response_features(observation: &Observation<'_>, decider: Seat) -> ResponseFeatures {
    let Some(response) = observation.response else {
        return ResponseFeatures {
            present: 0.0,
            tile: [0.0; TILE_FEATURES],
            relative_source: [0.0; PLAYER_COUNT],
            robbing_added_kong: 0.0,
        };
    };

    let mut tile = [0.0f32; TILE_KINDS];
    tile[response.tile.index() as usize] = 1.0;
    let mut relative_source = [0.0f32; PLAYER_COUNT];
    relative_source[decider.clockwise_distance(response.source) as usize] = 1.0;

    ResponseFeatures {
        present: 1.0,
        tile,
        relative_source,
        robbing_added_kong: flag(response.robbing_added_kong),
    }
}

fn seat_at_relative(origin: Seat, relative: usize) -> Seat {
    debug_assert!(relative < PLAYER_COUNT);
    let mut seat = origin;
    for _ in 0..relative {
        seat = seat.next_clockwise();
    }
    seat
}

fn player_view<'a>(observation: &'a Observation<'_>, seat: Seat) -> &'a PlayerView<'a> {
    observation
        .players
        .iter()
        .find(|view| view.seat == seat)
        .expect("observation covers every seat")
}

fn flag(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}
