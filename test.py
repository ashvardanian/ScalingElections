"""Cross-validates the Python, CUDA and Mojo implementations of the C2 voting rules.

Every expected answer comes from a brute-force oracle written straight from the definition of
the rule, never from the code under test, so a bug shared by all three ports cannot hide behind
their agreement. That bounds the sizes: the permutation oracles cap out near ten candidates, and
above them an exact integer program stands in.

Run with `uv run --no-sync pytest test.py`.
"""

import enum
import itertools
from collections.abc import Sequence

import numpy as np
import pytest

import ballots
import kemeny
import schulze
from conftest import derived_seed, exhaustive_scale, randomized_repetitions_count
from scalingelections import Backend, ScoreType, Unranked

# Ten ballots over Python, Rust, Go and Java, plus the voter who joins them ranking Java first.
PYTHON, RUST, GO, JAVA = range(4)
CANDIDATE_NAMES = ("Python", "Rust", "Go", "Java")
INVOLVEMENT_BALLOTS = [[0, 3, 1, 2]] + [[0, 3, 2, 1]] * 2 + [[1, 0, 2, 3]] * 2 + [[2, 3, 1, 0]] * 4 + [[3, 0, 1, 2]]
JAVA_FIRST_BALLOT = [3, 0, 2, 1]


# region Oracles


def oracle_pairwise_preferences(
    rankings: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Counts the ballots placing each candidate ahead of each other one, by definition."""
    num_candidates = max(max(ranking) for ranking in rankings) + 1
    preferences = np.zeros((num_candidates, num_candidates), dtype=np.uint32)
    for ranking in rankings:
        for position, preferred in enumerate(ranking):
            for opponent in ranking[position + 1 :]:
                preferences[preferred, opponent] += 1
    return preferences


def oracle_strongest_paths(preferences: np.ndarray) -> np.ndarray:
    """Widest paths over winning votes, by the textbook triple loop."""
    num_candidates = len(preferences)
    strengths = [[0] * num_candidates for _ in range(num_candidates)]
    for source in range(num_candidates):
        for target in range(num_candidates):
            if source != target and preferences[source][target] > preferences[target][source]:
                strengths[source][target] = int(preferences[source][target])
    for pivot in range(num_candidates):
        for source in range(num_candidates):
            for target in range(num_candidates):
                if pivot in (source, target) or source == target:
                    continue
                through_pivot = min(strengths[source][pivot], strengths[pivot][target])
                strengths[source][target] = max(strengths[source][target], through_pivot)
    return np.array(strengths, dtype=preferences.dtype).reshape(num_candidates, num_candidates)


def oracle_schulze_wins(preferences: np.ndarray) -> list[int]:
    """How many rivals each candidate beats on strongest paths, which is what Schulze ranks by."""
    strengths = oracle_strongest_paths(preferences)
    num_candidates = len(preferences)
    return [
        sum(1 for target in range(num_candidates) if strengths[source][target] > strengths[target][source])
        for source in range(num_candidates)
    ]


def kendall_score(preferences: np.ndarray, ranking: Sequence[int]) -> int:
    """Votes contradicting the ranking, summed over every pair it orders."""
    return sum(
        int(preferences[ranking[later]][ranking[earlier]])
        for earlier in range(len(ranking))
        for later in range(earlier + 1, len(ranking))
    )


def oracle_kemeny(preferences: np.ndarray) -> tuple[tuple[int, ...], int]:
    """The exact consensus ranking, by scoring every ordering there is."""
    orderings = itertools.permutations(range(len(preferences)))
    return min(
        ((order, kendall_score(preferences, order)) for order in orderings),
        key=lambda scored: scored[1],
    )


def oracle_split_cycle_winners(preferences: np.ndarray) -> list[int]:
    """
    The undefeated candidates under Holliday and Pacuit's Definition 3.3.

    A candidate defeats another when its margin is positive and strictly exceeds the smallest
    margin of every majority cycle carrying that edge, with the cycles enumerated rather than
    summarised by a widest path.
    """
    num_candidates = len(preferences)
    margins = [
        [int(preferences[source][target]) - int(preferences[target][source]) for target in range(num_candidates)]
        for source in range(num_candidates)
    ]
    defeated = set()
    for winner in range(num_candidates):
        for loser in range(num_candidates):
            if winner == loser or margins[winner][loser] <= 0:
                continue
            intermediates = [other for other in range(num_candidates) if other not in (winner, loser)]
            defeats = True
            for length in range(len(intermediates) + 1):
                for middle in itertools.permutations(intermediates, length):
                    cycle = (winner, loser, *middle, winner)
                    edges = [margins[cycle[step]][cycle[step + 1]] for step in range(len(cycle) - 1)]
                    if min(edges) <= 0:
                        continue  # Not a majority cycle, so it constrains nothing.
                    if margins[winner][loser] <= min(edges):
                        defeats = False
                        break
                if not defeats:
                    break
            if defeats:
                defeated.add(loser)
    return [candidate for candidate in range(num_candidates) if candidate not in defeated]


def tied_preferences(num_candidates: int, votes: int = 4) -> np.ndarray:
    """A profile where every pair is an exact tie, so no edge survives the winning-votes filter."""
    preferences = np.full((num_candidates, num_candidates), votes, dtype=np.uint32)
    np.fill_diagonal(preferences, 0)
    return preferences


def random_preferences(num_candidates: int, num_voters: int, seed: int) -> np.ndarray:
    """A reproducible profile drawn from random ballots."""
    return ballots.generate_preferences(num_candidates, num_voters, np.random.default_rng(seed))


BALLOT_SETS = {
    "involvement_10": INVOLVEMENT_BALLOTS,
    "involvement_11": [*INVOLVEMENT_BALLOTS, JAVA_FIRST_BALLOT],
    "condorcet_cycle": [[0, 1, 2]] * 3 + [[1, 2, 0]] * 3 + [[2, 0, 1]] * 3,
    "unanimous_four": [[0, 1, 2, 3]] * 5,
    "mirrored_four": [[0, 1, 2, 3]] * 3 + [[3, 2, 1, 0]] * 3,
}

PROFILES = {name: oracle_pairwise_preferences(rankings) for name, rankings in BALLOT_SETS.items()} | {
    "single_candidate": np.zeros((1, 1), dtype=np.uint32),
    "two_tied": tied_preferences(2),
    "all_tied_five": tied_preferences(5),
    "random_5": random_preferences(5, 9, derived_seed(5)),
    "random_6": random_preferences(6, 12, derived_seed(6)),
    "random_7": random_preferences(7, 15, derived_seed(7)),
    "random_8": random_preferences(8, 21, derived_seed(8)),
}

BALLOT_CASES = [pytest.param(rankings, id=name) for name, rankings in BALLOT_SETS.items()]
PROFILE_CASES = [pytest.param(matrix, id=name) for name, matrix in PROFILES.items()]


@pytest.mark.parametrize("rankings", BALLOT_CASES)
def test_pairwise_preferences_match_oracle(rankings: Sequence[Sequence[int]]):
    built = ballots.build_pairwise_preferences([np.asarray(ranking, dtype=np.int64) for ranking in rankings])
    assert np.array_equal(built, oracle_pairwise_preferences(rankings))


@pytest.mark.parametrize("preferences", PROFILE_CASES)
def test_kemeny_matches_oracle(preferences: np.ndarray):
    ranking, score, _, _ = kemeny.compute_kemeny_ranking(preferences)
    assert sorted(ranking) == list(range(len(preferences))), "The ranking must seat every candidate once"
    assert kendall_score(preferences, ranking) == score, "The reported ranking must achieve the reported score"
    assert score == oracle_kemeny(preferences)[1]


def test_kemeny_refuses_a_field_wider_than_the_table(implementation):
    """The cost table is exponential, so the width is refused before anything is allocated for it."""
    with pytest.raises(ValueError, match="33 candidates"):
        kemeny.compute_kemeny_ranking(np.zeros((36, 36), dtype=np.uint32))
    with pytest.raises(ValueError, match="33 candidates"):
        implementation.compute_kemeny_ranking(np.zeros((36, 36), dtype=np.uint32))


# endregion Oracles


# region Cross Language


@pytest.mark.parametrize("preferences", PROFILE_CASES)
def test_strongest_paths_agree_across_languages(preferences: np.ndarray, implementation):
    actual = np.asarray(implementation.compute_strongest_paths(preferences), dtype=np.uint32)
    assert np.array_equal(actual, oracle_strongest_paths(preferences))


@pytest.mark.parametrize("preferences", PROFILE_CASES)
def test_kemeny_agrees_across_languages(preferences: np.ndarray, implementation):
    expected = kemeny.compute_kemeny_ranking(preferences)
    actual = implementation.compute_kemeny_ranking(preferences)
    assert kendall_score(preferences, actual.ranking) == actual.score
    assert actual == expected


@pytest.mark.parametrize("preferences", PROFILE_CASES)
def test_margin_paths_feed_split_cycle_across_languages(preferences: np.ndarray, implementation):
    """The shared max-min kernel over margins must serve Split Cycle in every language."""
    margins = ballots.positive_margins(preferences)
    expected = oracle_split_cycle_winners(preferences)
    actual = np.asarray(implementation.compute_strongest_paths(margins), dtype=np.uint32)
    assert schulze.select_split_cycle_winners(preferences, actual) == expected


@pytest.mark.parametrize("preferences", PROFILE_CASES)
def test_split_cycle_agrees_across_languages(preferences: np.ndarray, implementation):
    """All three ports must name the same undefeated set, and it must match the definition."""
    expected = oracle_split_cycle_winners(preferences)
    assert list(implementation.compute_split_cycle_winners(preferences)) == expected


@pytest.mark.parametrize("preferences", PROFILE_CASES)
def test_split_cycle_contains_the_schulze_winners(preferences: np.ndarray, implementation):
    """The direct edge is itself a path, so every Schulze winner belongs to Split Cycle."""
    strengths = np.asarray(implementation.compute_strongest_paths(preferences), dtype=np.uint32)
    winners, _ = schulze.compute_election_results(list(range(len(preferences))), strengths)
    assert set(winners) <= set(implementation.compute_split_cycle_winners(preferences))


# endregion Cross Language


# region Schulze Backends

BACKEND_SIZES = (4, 8, 32, 64)


@pytest.mark.parametrize("num_candidates", BACKEND_SIZES)
def test_numba_parallel_matches_numba_serial(num_candidates: int, seed: int):
    preferences = random_preferences(num_candidates, 2 * num_candidates, seed + num_candidates)
    expected = schulze.compute_strongest_paths_serial(preferences)
    assert np.array_equal(schulze.compute_strongest_paths_tiled_cpu(preferences), expected)


# endregion Schulze Backends


# region Edge Cases


@pytest.mark.parametrize("num_candidates", (2, 5, schulze.TILE_SIZE + 1))
def test_tie_heavy_profiles_leave_no_edges(num_candidates: int, implementation):
    preferences = tied_preferences(num_candidates)
    strengths = implementation.compute_strongest_paths(preferences)
    assert not strengths.any(), "A tie beats nobody, so no path can carry strength"
    assert np.array_equal(strengths, oracle_strongest_paths(preferences))


def test_tie_heavy_profiles_keep_every_candidate():
    preferences = PROFILES["mirrored_four"]
    margin_paths = schulze.compute_strongest_paths_serial(ballots.positive_margins(preferences))
    assert schulze.select_split_cycle_winners(preferences, margin_paths) == list(range(len(preferences)))


@pytest.mark.parametrize("num_candidates", (2, 4))
def test_kemeny_solves_a_disagreement_wider_than_thirty_two_bits(implementation, num_candidates):
    """The UInt32 sentinel and scores above it both require 64-bit arithmetic."""
    preferences = np.full((num_candidates, num_candidates), np.iinfo(np.uint32).max, dtype=np.uint32)
    np.fill_diagonal(preferences, 0)
    expected = kemeny.compute_kemeny_ranking(preferences)
    python_ranking, python_score, _, _ = expected
    assert python_score >= np.iinfo(np.uint32).max
    assert kendall_score(preferences, python_ranking) == python_score
    assert implementation.compute_kemeny_ranking(preferences) == expected
    with pytest.raises(OverflowError, match="arithmetic type"):
        implementation.compute_kemeny_ranking(preferences, score_type=ScoreType.uint32)
    with pytest.raises(OverflowError, match="arithmetic type"):
        kemeny.compute_kemeny_ranking(preferences, score_type=ScoreType.uint32)


def test_kemeny_score_widths_preserve_the_optimum(implementation):
    preferences = PROFILES["random_7"].copy()
    np.fill_diagonal(preferences, np.iinfo(np.uint32).max)
    expected_score = oracle_kemeny(preferences)[1]
    for score_type in ScoreType:
        expected = kemeny.compute_kemeny_ranking(preferences, score_type=score_type)
        result = implementation.compute_kemeny_ranking(preferences, score_type=score_type)
        assert result == expected
        assert result.score == expected_score
    with pytest.raises(ValueError, match="uint24"):
        implementation.compute_kemeny_ranking(preferences, score_type="uint24")


@pytest.mark.slow
def test_kemeny_solves_a_national_electorate(implementation):
    """A 350-million-voter field of twenty candidates, which a 32-bit score refused outright."""
    num_candidates, half_the_electorate = 20, 175_000_000
    preferences = np.zeros((num_candidates, num_candidates), dtype=np.uint32)
    # Every pair split down the middle, so no ordering escapes paying half the electorate per pair.
    preferences[:] = half_the_electorate
    np.fill_diagonal(preferences, 0)
    expected = kemeny.compute_kemeny_ranking(preferences)
    python_ranking, python_score, _, _ = expected
    assert python_score > np.iinfo(np.uint32).max
    assert kendall_score(preferences, python_ranking) == python_score
    assert implementation.compute_kemeny_ranking(preferences) == expected


# endregion Edge Cases


# region Paradoxes


def test_representative_ranking_does_not_hide_tied_schulze_winners():
    before, after = PROFILES["involvement_10"], PROFILES["involvement_11"]
    candidates = list(range(len(CANDIDATE_NAMES)))
    winners_before, ranking_before = schulze.compute_election_results(
        candidates, schulze.compute_strongest_paths_serial(before)
    )
    winners_after, ranking_after = schulze.compute_election_results(
        candidates, schulze.compute_strongest_paths_serial(after)
    )
    assert (ranking_before[0], ranking_after[0]) == (JAVA, PYTHON)
    assert winners_before == [PYTHON, GO, JAVA]
    assert winners_after == [PYTHON, JAVA]


def test_split_cycle_keeps_the_supported_candidate_among_tied_winners():
    """Split Cycle satisfies positive involvement on the same profile: Java stays in the winner set."""
    before, after = PROFILES["involvement_10"], PROFILES["involvement_11"]
    winners = []
    for preferences in (before, after):
        margin_paths = schulze.compute_strongest_paths_serial(ballots.positive_margins(preferences))
        computed = schulze.select_split_cycle_winners(preferences, margin_paths)
        assert computed == oracle_split_cycle_winners(preferences)
        winners.append(computed)
    assert winners[0] == [PYTHON, GO, JAVA]
    assert winners[1] == [PYTHON, JAVA]
    assert JAVA in winners[0] and JAVA in winners[1], "The extra Java-first ballot must not unseat Java"


DIVERGENT_BALLOTS = [
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [3, 0, 2, 1],
    [2, 1, 3, 0],
    [1, 0, 2, 3],
    [2, 3, 1, 0],
    [3, 0, 2, 1],
    [2, 1, 3, 0],
    [3, 0, 2, 1],
]


def test_the_three_rules_diverge_on_one_electorate(implementation):
    """Nine ballots where each rule answers differently, which the README publishes."""
    preferences = oracle_pairwise_preferences(DIVERGENT_BALLOTS)
    assert not any(
        all(preferences[source][target] > preferences[target][source] for target in range(4) if target != source)
        for source in range(4)
    ), "the profile has to cycle, or every rule would agree"

    # Schulze names the candidate no other candidate beats on path strength.
    strengths = oracle_strongest_paths(preferences)
    undominated = [
        source
        for source in range(4)
        if all(strengths[source][target] >= strengths[target][source] for target in range(4) if target != source)
    ]
    assert undominated == [JAVA]

    assert oracle_split_cycle_winners(preferences) == [RUST, GO, JAVA]

    ranking, score, _, _ = implementation.compute_kemeny_ranking(preferences)
    assert list(ranking) == [GO, RUST, JAVA, PYTHON]
    assert score == kendall_score(preferences, ranking)
    runner_up = sorted(kendall_score(preferences, order) for order in itertools.permutations(range(4)))[1]
    assert score < runner_up, "the optimum must be decisive rather than tie-broken"


# endregion Paradoxes


# region Differential


class BallotSpread(enum.Enum):
    """How much the electorate agrees, which is what decides whether tie-breaking is exercised."""

    distinct = enum.auto()
    half_replayed = enum.auto()
    mirrored = enum.auto()


SPREAD_CASES = [pytest.param(spread, id=spread.name) for spread in BallotSpread]


def random_profile(
    seed: int,
    num_candidates: int,
    num_voters: int,
    spread: BallotSpread = BallotSpread.distinct,
) -> np.ndarray:
    """Tallies random ballots, so the matrix is always realizable rather than an arbitrary grid."""
    generator = np.random.default_rng(seed)
    preferences = np.zeros((num_candidates, num_candidates), dtype=np.uint32)
    for ballot in range(num_voters):
        ranking = generator.permutation(num_candidates)
        # Replaying one canonical order for half the electorate manufactures the ties that
        # separate implementations agreeing on a score from implementations agreeing on a ranking.
        if spread is BallotSpread.half_replayed and ballot % 2:
            ranking = np.arange(num_candidates)
        ballots.populate_preferences_from_ranking(preferences, ranking)
        # Answering every ballot with its reverse ties every pair exactly, which is where
        # tie-breaking has to be identical across the ports rather than merely equally scored.
        if spread is BallotSpread.mirrored:
            ballots.populate_preferences_from_ranking(preferences, ranking[::-1].copy())
    return preferences


# Sizes straddling the compile-time tile, where a mishandled tail is invisible on round numbers.
TILE_BOUNDARY_SIZES = (1, 2, 31, 32, 33, 47, 63, 64, 65, 96, 97, 129)


@pytest.mark.parametrize("num_candidates", TILE_BOUNDARY_SIZES)
def test_backends_agree_across_tile_boundaries(num_candidates: int, seed: int, implementation):
    """Every backend must match the serial baseline whether or not the tile divides the electorate."""
    preferences = random_profile(seed=seed + num_candidates, num_candidates=num_candidates, num_voters=25)
    expected = schulze.compute_strongest_paths_serial(preferences)
    assert np.array_equal(schulze.compute_strongest_paths_tiled_cpu(preferences), expected)
    actual = np.asarray(implementation.compute_strongest_paths(preferences), dtype=np.uint32)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("step", range(randomized_repetitions_count))
def test_languages_agree_on_random_profiles(step: int, seed: int, implementation):
    """Schulze and Kemeny must agree across all three ports on profiles nobody chose by hand."""
    num_candidates = 2 + step % 9
    spread = BallotSpread.half_replayed if step % 3 == 0 else BallotSpread.distinct
    preferences = random_profile(
        seed=seed + step,
        num_candidates=num_candidates,
        num_voters=1 + step * 3,
        spread=spread,
    )

    expected = schulze.compute_strongest_paths_serial(preferences)
    assert np.array_equal(
        np.asarray(implementation.compute_strongest_paths(preferences), dtype=np.uint32),
        expected,
    )

    # Kemeny is exact, so the ranking has to match and not merely the score it achieves.
    python_ranking, python_score, _, _ = kemeny.compute_kemeny_ranking(preferences)
    actual_ranking, actual_score, _, _ = implementation.compute_kemeny_ranking(preferences)
    assert list(python_ranking) == list(actual_ranking)
    assert int(python_score) == int(actual_score)
    assert int(python_score) == kendall_score(preferences, python_ranking)


@pytest.mark.parametrize("step", range(randomized_repetitions_count))
def test_schulze_winners_always_inside_split_cycle(step: int, seed: int):
    """The direct edge is itself a path, so every Schulze winner belongs to Split Cycle."""
    num_candidates = 3 + step % 8
    preferences = random_profile(seed=seed + step, num_candidates=num_candidates, num_voters=2 + step * 2)
    strongest = schulze.compute_strongest_paths_serial(preferences)
    winners, _ = schulze.compute_election_results(list(range(num_candidates)), strongest)
    margin_paths = schulze.compute_strongest_paths_serial(ballots.positive_margins(preferences))
    assert set(winners) <= set(schulze.select_split_cycle_winners(preferences, margin_paths))


# endregion Differential


# region Kemeny Backends


KEMENY_DIFFERENTIAL_SIZES = tuple(14 + 2 * rung for rung in range(1 + exhaustive_scale))


@pytest.mark.slow
@pytest.mark.parametrize("spread", SPREAD_CASES)
@pytest.mark.parametrize("num_candidates", KEMENY_DIFFERENTIAL_SIZES)
def test_kemeny_backends_agree_on_larger_fields(num_candidates: int, spread: BallotSpread, seed: int, implementation):
    preferences = random_profile(
        seed=seed + num_candidates,
        num_candidates=num_candidates,
        num_voters=30,
        spread=spread,
    )
    ranking, score, _, _ = implementation.compute_kemeny_ranking(preferences)
    expected_ranking, expected_score, _, _ = kemeny.compute_kemeny_ranking(preferences)
    assert list(ranking) == expected_ranking
    assert score == expected_score == kendall_score(preferences, ranking)


# endregion Kemeny Backends


# region Narrow Sweep


# The packed sweep holds two candidates per word, so it is only reachable below this ceiling.
SIXTEEN_BIT_CEILING = 65535


def scaled_profile(num_candidates: int, peak: int, seed: int) -> np.ndarray:
    """Tallies real ballots, then scales them so the largest count lands exactly on `peak`."""
    preferences = random_profile(seed=seed, num_candidates=num_candidates, num_voters=40)
    largest = int(preferences.max())
    scaled = (preferences.astype(np.uint64) * (peak // largest)).astype(np.uint32)
    scaled[preferences == largest] = peak
    return scaled


@pytest.mark.parametrize("peak", (SIXTEEN_BIT_CEILING, SIXTEEN_BIT_CEILING + 1, 4_200_000_000))
@pytest.mark.parametrize("num_candidates", (33, 64))
def test_backends_agree_across_the_sixteen_bit_boundary(num_candidates: int, peak: int, seed: int, implementation):
    """One vote count above the ceiling has to move the whole sweep to the wide path, not truncate."""
    preferences = scaled_profile(num_candidates, peak, seed=seed + num_candidates)
    expected = schulze.compute_strongest_paths_serial(preferences)
    assert np.array_equal(implementation.compute_strongest_paths(preferences), expected)


# endregion Narrow Sweep


# region Tally


@pytest.mark.parametrize("rankings", BALLOT_CASES)
def test_tally_matches_the_oracle(rankings: Sequence[Sequence[int]], implementation):
    """Counted ballots must equal the definition, counted by hand."""
    expected = oracle_pairwise_preferences(rankings)
    built = implementation.tally_ballots(np.asarray(rankings, dtype=np.uint32))
    assert np.array_equal(built, expected)


def test_tally_in_chunks_equals_tally_in_one(seed: int):
    """Splitting the electorate must not change the count, since that is what lets it stream."""
    num_candidates = 40
    generator = np.random.default_rng(seed)
    rankings = np.array([generator.permutation(num_candidates) for _ in range(900)], dtype=np.uint32)
    whole = ballots.tally_chunks([rankings], num_candidates)
    chunked = ballots.tally_chunks(
        (rankings[start : start + 128] for start in range(0, len(rankings), 128)),
        num_candidates,
    )
    assert np.array_equal(whole, chunked)


def test_tally_omitted_candidates_are_not_ordered_by_index():
    counted = ballots.build_pairwise_preferences([[2, 0]], num_candidates=4, unranked=Unranked.worse)
    assert counted[2, 0] == 1 and counted[0, 1] == 1 and counted[0, 3] == 1
    assert counted[1, 3] == counted[3, 1] == 0
    unknown = ballots.build_pairwise_preferences([[2, 0]], num_candidates=4)
    assert unknown.sum() == unknown[2, 0] == 1


@pytest.mark.parametrize("implementation", ["cpp-gpu"], indirect=True)
def test_cpp_gpu_tally_capacity(implementation):
    limit = 106
    rankings = np.arange(limit, dtype=np.uint32).reshape(1, limit)
    assert np.array_equal(implementation.tally_ballots(rankings), oracle_pairwise_preferences(rankings))
    with pytest.raises(ValueError):
        implementation.tally_ballots(np.arange(limit + 1, dtype=np.uint32).reshape(1, limit + 1))


@pytest.mark.parametrize("implementation", ["mojo-cpu", "mojo-gpu"], indirect=True)
def test_mojo_tally_exceeds_dense_kernel_capacity(implementation):
    rankings = np.arange(65, dtype=np.uint32).reshape(1, 65)
    assert np.array_equal(implementation.tally_ballots(rankings), oracle_pairwise_preferences(rankings))


# endregion Tally


# region Third Party


def as_profile(rankings: Sequence[Sequence[int]]):
    """The same ballots in the shape the reference library expects."""
    from pref_voting.profiles import Profile

    return Profile([[int(candidate) for candidate in ranking] for ranking in rankings])


def oracle_kemeny_score_by_arc_set(preferences: np.ndarray) -> int:
    """
    The exact Kemeny score as a minimum-weight feedback arc set, solved by integer program.

    An ordering pays the weight of every pair it puts backwards, so the least any ranking can
    disagree is the lightest set of arcs whose removal leaves the pairwise graph acyclic. This
    reaches field widths the permutation oracle cannot, `ip_ti` being exact rather than greedy.
    """
    import igraph

    num_candidates = len(preferences)
    edges, weights = [], []
    for source in range(num_candidates):
        for target in range(num_candidates):
            if source != target and preferences[source][target]:
                edges.append((source, target))
                weights.append(int(preferences[source][target]))
    graph = igraph.Graph(n=num_candidates, edges=edges, directed=True)
    return sum(weights[arc] for arc in graph.feedback_arc_set(weights=weights, method="ip_ti"))


# Wide enough that enumerating orderings is hopeless, which is the range the published tables cover.
KEMENY_ORACLE_SIZES = tuple(21 + 3 * rung for rung in range(exhaustive_scale))


@pytest.mark.parametrize("preferences", PROFILE_CASES)
def test_the_two_kemeny_oracles_agree(preferences: np.ndarray, igraph):
    """The integer program has to reproduce what enumerating every ordering already proves."""
    assert oracle_kemeny_score_by_arc_set(preferences) == oracle_kemeny(preferences)[1]


@pytest.mark.slow
@pytest.mark.parametrize("num_candidates", KEMENY_ORACLE_SIZES)
def test_kemeny_matches_the_arc_set_oracle_beyond_enumeration(num_candidates: int, seed: int, igraph, implementation):
    """Past the permutation oracle's reach, an exact integer program keeps the subset search honest."""
    preferences = random_profile(seed=seed + num_candidates, num_candidates=num_candidates, num_voters=25)
    expected = oracle_kemeny_score_by_arc_set(preferences)
    python_ranking, python_score, _, _ = kemeny.compute_kemeny_ranking(preferences)
    assert kendall_score(preferences, python_ranking) == python_score
    assert python_score == expected
    assert int(implementation.compute_kemeny_ranking(preferences)[1]) == expected


@pytest.mark.parametrize("rankings", BALLOT_CASES)
def test_split_cycle_matches_the_reference_library(rankings: Sequence[Sequence[int]], pref_voting, implementation):
    """Our Split Cycle must name the set its own authors' implementation names."""
    from pref_voting.margin_based_methods import split_cycle

    preferences = oracle_pairwise_preferences(rankings)
    assert list(implementation.compute_split_cycle_winners(preferences)) == sorted(split_cycle(as_profile(rankings)))


@pytest.mark.parametrize("rankings", BALLOT_CASES)
def test_schulze_matches_the_reference_library(rankings: Sequence[Sequence[int]], pref_voting, implementation):
    """Schulze is Beat Path under another name, so the undominated sets must coincide."""
    from pref_voting.margin_based_methods import beat_path

    preferences = oracle_pairwise_preferences(rankings)
    strengths = implementation.compute_strongest_paths(preferences)
    count = len(preferences)
    undominated = [
        source
        for source in range(count)
        if all(strengths[source][target] >= strengths[target][source] for target in range(count) if target != source)
    ]
    assert undominated == sorted(beat_path(as_profile(rankings)))


@pytest.mark.parametrize("rankings", BALLOT_CASES)
def test_kemeny_matches_the_reference_library(rankings: Sequence[Sequence[int]], pref_voting, implementation):
    """Our winner set must match the reference library, ties included."""
    from pref_voting.other_methods import kemeny_young

    preferences = oracle_pairwise_preferences(rankings)
    result = implementation.compute_kemeny_ranking(preferences)
    assert result.winners == sorted(kemeny_young(as_profile(rankings)))


@pytest.mark.parametrize("step", range(randomized_repetitions_count))
def test_all_three_match_the_reference_library_on_random_profiles(step: int, seed: int, pref_voting, implementation):
    """The agreement has to hold off the hand-picked profiles too."""
    from pref_voting.margin_based_methods import beat_path, split_cycle
    from pref_voting.other_methods import kemeny_young

    generator = np.random.default_rng(seed + step)
    num_candidates = 3 + step % 4
    rankings = [list(generator.permutation(num_candidates)) for _ in range(3 + step)]
    preferences = oracle_pairwise_preferences(rankings)
    profile = as_profile(rankings)

    assert list(implementation.compute_split_cycle_winners(preferences)) == sorted(split_cycle(profile))
    strengths = implementation.compute_strongest_paths(preferences)
    undominated = [
        source
        for source in range(num_candidates)
        if all(
            strengths[source][target] >= strengths[target][source]
            for target in range(num_candidates)
            if target != source
        )
    ]
    assert undominated == sorted(beat_path(profile))
    result = implementation.compute_kemeny_ranking(preferences)
    assert result.winners == sorted(kemeny_young(profile))


# endregion Third Party


def test_dispatch_rejects_unknown_options(implementation):
    for operation in (
        implementation.tally_ballots,
        implementation.compute_strongest_paths,
        implementation.compute_kemeny_ranking,
        implementation.compute_split_cycle_winners,
    ):
        with pytest.raises(ValueError, match="backend"):
            operation([[0]], backend="metal_typo")
        with pytest.raises(TypeError):
            operation([[0]], backned="cpu")


@pytest.mark.parametrize("num_candidates", (33, 64))
def test_tally_crosses_word_boundaries(num_candidates: int, seed: int, implementation):
    generator = np.random.default_rng(seed)
    rankings = np.array([generator.permutation(num_candidates) for _ in range(300)], dtype=np.uint32)
    assert np.array_equal(implementation.tally_ballots(rankings), oracle_pairwise_preferences(rankings))


def test_empty_ballots_preserve_the_candidate_count(implementation):
    assert np.array_equal(
        implementation.tally_ballots(np.empty((0, 3), dtype=np.uint32)),
        np.zeros((3, 3), dtype=np.uint32),
    )


def test_counts_are_validated_before_narrowing(implementation):
    for invalid in (-1, 1 << 64):
        with pytest.raises(OverflowError):
            implementation.compute_strongest_paths([[0, invalid], [0, 0]])


def test_mojo_matrix_buffers_preserve_layout_and_wide_counts():
    module = pytest.importorskip("scalingelections_mojo")
    expected = np.array([[0, (1 << 63) + 1], [0, 0]], dtype=np.uint64)
    storage = np.zeros((4, 4), dtype=np.uint64)
    storage[::2, ::2] = expected
    strided = storage[::2, ::2]
    strided.flags.writeable = False
    readonly = expected.copy()
    readonly.flags.writeable = False
    for preferences in (readonly, strided, expected.astype(">u8"), expected.tolist()):
        result = module.compute_strongest_paths(preferences)
        assert result.dtype == np.uint64
        assert result.flags.c_contiguous
        assert np.array_equal(result, expected)


def test_cpp_dense_ranks_must_match_both_dimensions():
    module = pytest.importorskip("scalingelections_cuda")
    rankings = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint32)
    with pytest.raises(ValueError, match="Ranks must match"):
        module.tally_ballots(rankings, ranks=np.zeros((3, 2), dtype=np.uint32))


def test_kemeny_auto_width_reserves_the_unreachable_sentinel():
    preferences = np.array([[0, np.iinfo(np.uint32).max - 1], [0, 0]], dtype=np.uint32)
    assert kemeny.resolve_score_type(preferences) == ScoreType.uint32
    preferences[0, 1] += 1
    assert kemeny.resolve_score_type(preferences) == ScoreType.uint64


def test_explicit_gpu_request_does_not_fall_back_to_python():
    with pytest.raises(RuntimeError, match="gpu"):
        ballots.tally_chunks([], 3, backend=Backend.gpu, implementation="python")


def test_schulze_score_types_preserve_paths(implementation):
    preferences = np.array([[0, 65535, 0], [0, 0, 65534], [65533, 0, 0]], dtype=np.uint32)
    expected = oracle_strongest_paths(preferences)
    for score_type in ScoreType:
        paths = implementation.compute_strongest_paths(preferences, score_type=score_type)
        assert np.array_equal(paths, expected)
        assert paths.dtype == np.dtype(f"uint{32 if score_type is ScoreType.auto else score_type.bits}")
        assert list(implementation.compute_split_cycle_winners(preferences, score_type=score_type)) == [0]
    shifted = preferences + np.uint32(65536)
    np.fill_diagonal(shifted, 0)
    assert list(implementation.compute_split_cycle_winners(shifted, score_type=ScoreType.uint16)) == [0]
    preferences[0, 1] += 1
    with pytest.raises(OverflowError):
        implementation.compute_strongest_paths(preferences, score_type=ScoreType.uint16)


def test_weighted_csr_ballots_preserve_ties_and_omissions(implementation):
    options = dict(
        offsets=[0, 3, 5, 5],
        num_candidates=4,
        ranks=[0, 0, 1, 0, 0],
        weights=[3, 5, 7],
    )
    ids = [0, 1, 2, 1, 2]
    unknown = np.array([[0, 0, 3, 0], [0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint64)
    worse = np.array([[0, 0, 3, 3], [5, 0, 3, 8], [5, 0, 0, 8], [0, 0, 0, 0]], dtype=np.uint64)
    assert np.array_equal(implementation.tally_ballots(ids, **options, unranked=Unranked.unknown), unknown)
    assert np.array_equal(implementation.tally_ballots(ids, **options, unranked=Unranked.worse), worse)


def test_weighted_counts_preserve_uint64_through_solvers(implementation):
    preferences = implementation.tally_ballots([[0, 1]], weights=[1 << 40])
    assert preferences[0, 1] == 1 << 40
    assert np.array_equal(implementation.compute_strongest_paths(preferences), preferences)
    assert implementation.compute_split_cycle_winners(preferences) == [0]
    assert implementation.compute_kemeny_ranking(preferences) == ([0, 1], 0, [0], True)


def test_tied_results_report_all_winners_and_nonunique_order(implementation):
    preferences = tied_preferences(3)
    paths = implementation.compute_strongest_paths(preferences)
    winners, ranking = schulze.compute_election_results([0, 1, 2], paths)
    assert winners == [0, 1, 2]
    assert sorted(ranking) == winners
    result = implementation.compute_kemeny_ranking(preferences)
    assert result.winners == winners
    assert not result.unique
    assert result.score == 12
    preferences = np.array([[0, 3, 3], [0, 0, 1], [0, 1, 0]], dtype=np.uint64)
    result = implementation.compute_kemeny_ranking(preferences)
    assert result.winners == [0]
    assert not result.unique


def test_saturated_kemeny_discards_overflowing_alternatives(implementation):
    preferences = np.triu(np.full((3, 3), 1 << 63, dtype=np.uint64), k=1)
    result = implementation.compute_kemeny_ranking(preferences, score_type=ScoreType.saturated64)
    assert result == ([0, 1, 2], 0, [0], True)
    assert implementation.compute_kemeny_ranking(preferences) == result
    preferences += preferences.T
    with pytest.raises(OverflowError):
        implementation.compute_kemeny_ranking(preferences, score_type=ScoreType.saturated64)


def test_saturated_tally_rejects_only_overflowing_cells(implementation):
    weight = (1 << 64) - 2
    preferences = implementation.tally_ballots(
        [[0, 1], [1, 0]], weights=[weight, weight], score_type=ScoreType.saturated64
    )
    assert int(preferences[0, 1]) == int(preferences[1, 0]) == weight
    for increment in (1, 2):
        with pytest.raises(OverflowError):
            implementation.tally_ballots(
                [[0, 1], [0, 1]], weights=[weight, increment], score_type=ScoreType.saturated64
            )


def test_csr_ballots_reject_duplicate_ids_and_invalid_offsets(implementation):
    with pytest.raises(ValueError):
        implementation.tally_ballots([0, 0], offsets=[0, 2], num_candidates=2)
    with pytest.raises(ValueError):
        implementation.tally_ballots([0, 1], offsets=[0, 3, 2], num_candidates=2)


@pytest.mark.parametrize("module_name", ["scalingelections_cuda", "scalingelections_mojo"])
def test_native_saturation_boundaries(module_name):
    module = pytest.importorskip(module_name)
    maximum = (1 << 64) - 1
    for backend in module.available_backends():
        for count in (maximum - 1, maximum):
            preferences = np.array([[0, count], [count, 0]], dtype=np.uint64)
            if count == maximum:
                with pytest.raises(OverflowError):
                    module.compute_kemeny_ranking(preferences, backend=backend, score_type="saturated64")
                with pytest.raises(OverflowError):
                    module.compute_strongest_paths(preferences, backend=backend, score_type="saturated64")
            else:
                assert module.compute_kemeny_ranking(preferences, backend=backend, score_type="saturated64")[1] == count
            if count == maximum:
                with pytest.raises(OverflowError):
                    module.tally_ballots([[0, 1]], weights=[count], backend=backend, score_type="saturated64")
            else:
                result = module.tally_ballots([[0, 1]], weights=[count], backend=backend, score_type="saturated64")
                assert result.dtype == np.uint64
                assert int(result[0, 1]) == count


def test_tally_output_uses_selected_width(implementation):
    for score_type in (ScoreType.uint16, ScoreType.uint32, ScoreType.uint64, ScoreType.saturated64):
        for weights in (None, [1]):
            result = implementation.tally_ballots([[0, 1]], weights=weights, score_type=score_type)
            assert result.dtype == np.dtype(f"uint{score_type.bits}")
            assert int(result[0, 1]) == 1
