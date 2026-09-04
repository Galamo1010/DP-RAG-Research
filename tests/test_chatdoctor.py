"""Tests for the reference-damage heuristic (pure function, no dataset needed).

The quality scores are computed against iCliniq's doctor replies, and those
replies are damaged by the dataset's own construction. `reference_damage` is what
lets the damaged and undamaged subsets be scored separately, so that the report
can say the contamination was measured rather than assumed.

The measurement it reproduces is in `docs/notes/reference-damage.md`. That table
was produced ad hoc and the detector was not kept, so the two do not agree
exactly: over the 200-query sample this version finds 34 truncated-only, 24
attachment-only, 4 both and 138 undamaged, against the note's 33 / 25 / 3 / 139 --
within one in every cell, the difference being the function-word list, which the
original run never recorded. That is itself the lesson the note draws: a
measurement whose code is not kept is not a measurement you still have.
"""

from dprag.chatdoctor import ATTACHMENT, TRUNCATED, reference_damage


# --------------------------------------------------------------------------
# Truncation -- content cut mid-sentence with "ChatDoctor" spliced in
# --------------------------------------------------------------------------

def test_a_half_eaten_drug_name_is_truncation():
    """The case that motivates the whole check: the reference lost the drug."""
    assert reference_damage("Take Sudafed (Pseudoephe ChatDoctor.") == {TRUNCATED}


def test_a_cut_mid_word_is_truncation():
    assert reference_damage("Do you have a dan ChatDoctor.") == {TRUNCATED}


def test_a_cut_after_a_whole_content_word_is_truncation():
    assert reference_damage("nothing to do with anesthesia ChatDoctor.") == {TRUNCATED}


# --------------------------------------------------------------------------
# The false positives the function-word list exists to prevent
# --------------------------------------------------------------------------

def test_naming_the_platform_normally_is_not_damage():
    """These sentences are intact; the platform is simply mentioned. Counting
    them would inflate the damage rate with ordinary text."""
    assert reference_damage("Welcome to ChatDoctor forum.") == set()
    assert reference_damage("Thanks for consulting ChatDoctor.") == set()


def test_the_check_is_case_insensitive_on_the_function_word():
    assert reference_damage("Welcome To ChatDoctor.") == set()


def test_clean_text_carries_no_damage():
    assert reference_damage("Drink plenty of water and rest for two days.") == set()


# --------------------------------------------------------------------------
# Attachment -- the reference is intact but the task is impossible
# --------------------------------------------------------------------------

def test_attachment_removed_is_its_own_kind():
    """Distinct from truncation: nothing is missing from the TEXT, but the doctor
    is describing an image the model never sees, so the ceiling is unreachable."""
    damage = reference_damage(
        "As per your MRI (attachment removed to protect patient identity), "
        "the disc is bulging."
    )
    assert damage == {ATTACHMENT}


def test_the_shorter_attachment_wording_also_matches():
    assert ATTACHMENT in reference_damage("(attachment removed to protect identity)")


# --------------------------------------------------------------------------
# Both, and the empty case
# --------------------------------------------------------------------------

def test_a_reference_can_carry_both_kinds():
    """They are returned as a set rather than one label because they depress a
    score for different reasons and may not affect a semantic metric equally."""
    damage = reference_damage(
        "Your scan (attachment removed to protect patient identity) shows "
        "inflammation, so start azithro ChatDoctor."
    )
    assert damage == {TRUNCATED, ATTACHMENT}


def test_empty_input_is_undamaged_rather_than_raising():
    assert reference_damage("") == set()
