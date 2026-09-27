"""What has to hold for a canary leakage number to mean anything.

Each test pins one failure that would otherwise pass silently and produce a
plausible-looking zero -- the most expensive outcome available to this probe,
because "the detector is broken" and "differential privacy worked" look identical
in the output.
"""

from __future__ import annotations

import pytest

from dprag import canary


def test_generate_is_deterministic():
    """Same seed, same canaries -- the run record stores the seed, not the secrets."""
    a_inj, a_hold = canary.generate(50, seed=1)
    b_inj, b_hold = canary.generate(50, seed=1)
    assert [c.to_dict() for c in a_inj] == [c.to_dict() for c in b_inj]
    assert [c.to_dict() for c in a_hold] == [c.to_dict() for c in b_hold]


def test_different_seeds_give_different_canaries():
    a, _ = canary.generate(50, seed=1)
    b, _ = canary.generate(50, seed=2)
    assert {c.surname for c in a}.isdisjoint({c.surname for c in b})


def test_surnames_unique_across_both_halves():
    """A surname shared between planted and holdout makes a hit unattributable,
    which destroys the one distinction the holdout exists to draw."""
    inj, hold = canary.generate(500, seed=3)
    names = [c.surname.lower() for c in inj + hold]
    assert len(set(names)) == len(names)


def test_holdout_is_marked_and_equal_sized():
    inj, hold = canary.generate(40, seed=4)
    assert len(inj) == len(hold) == 40
    assert all(c.injected for c in inj)
    assert not any(c.injected for c in hold)


def test_name_length_floors():
    """Short names collide with ordinary English. The first version drew
    one-syllable given names ("Rim", "Zu", "Son") and the detector then fired on
    19 of 300 un-injected real documents."""
    inj, hold = canary.generate(300, seed=5)
    for c in inj + hold:
        assert len(c.surname) >= canary.MIN_SURNAME, c.surname
        assert len(c.given) >= canary.MIN_GIVEN, c.given


def test_every_canary_has_name_and_age():
    inj, _ = canary.generate(100, seed=6)
    for c in inj:
        assert c.surname and c.given
        assert 0 < c.age < 120


def test_optional_fields_actually_vary():
    """If every canary carried every field, the report could not say WHICH kind of
    identifier survives -- and the format would be a template the model can
    complete from position alone."""
    inj, _ = canary.generate(200, seed=7)
    for field in ("tel", "email", "allergy", "history"):
        present = sum(1 for c in inj if getattr(c, field))
        assert 0 < present < len(inj), f"{field} is not varying: {present}/200"


def test_inject_prepends_and_preserves_the_document():
    inj, _ = canary.generate(3, seed=8)
    docs = ["alpha text", "beta text", "gamma text"]
    out = canary.inject(docs, inj)
    for canary_obj, original, result in zip(inj, docs, out):
        assert result.startswith(canary_obj.prefix())
        assert result.endswith(original)


def test_inject_refuses_length_mismatch():
    """Zipping to the shorter list would leave documents un-injected and make the
    leakage denominator wrong in a way nothing downstream could detect."""
    inj, _ = canary.generate(3, seed=9)
    with pytest.raises(ValueError, match="one canary per document"):
        canary.inject(["only one document"], inj)


def test_detector_finds_every_present_field():
    inj, hold = canary.generate(20, seed=10)
    det = canary.Detector(inj + hold)
    # A canary that carries every optional field, so nothing is skipped by luck.
    c = next(x for x in inj if all(getattr(x, f)
                                   for f in ("tel", "email", "allergy", "history")))
    hits = det.find(canary.inject(["body text"], [c])[0])
    assert c.cid in hits["surname"]
    for field in ("tel", "email", "allergy", "history"):
        assert c.cid in hits[field], f"missed {field}"


def test_detector_is_silent_on_text_with_no_canary():
    """A detector that fires on ordinary prose inflates the holdout, and the
    holdout is the number that licenses reading everything else."""
    inj, hold = canary.generate(200, seed=11)
    det = canary.Detector(inj + hold)
    prose = (
        "Hello, welcome to Chat Doctor. Premature ejaculation is a very common "
        "problem. Take 500 mg twice daily for 7 years and consult your physician. "
        "Your wife should also be examined. Call 150-300 for blood pressure."
    )
    assert det.find(prose) == {}


def test_detector_does_not_key_on_given_name():
    """Given names are not unique and are short enough to be real words, so a hit
    on one identifies nothing. Keying on it was a measured source of false
    positives."""
    assert "given" not in canary.FIELDS
    inj, hold = canary.generate(20, seed=12)
    det = canary.Detector(inj + hold)
    assert det.find(f"The patient {inj[0].given} came in today.") == {}


def test_split_separates_planted_from_holdout():
    inj, hold = canary.generate(10, seed=13)
    det = canary.Detector(inj + hold)
    text = f"{inj[0].surname} and {hold[0].surname} both appear here."
    planted_hits, holdout_hits = det.split(det.find(text))
    assert planted_hits["surname"] == {inj[0].cid}
    assert holdout_hits["surname"] == {hold[0].cid}


def test_collisions_uses_the_detector_matching_rule():
    """A check stricter than the instrument it guards rejects good seeds. An
    earlier version used substring containment and refused on a canary whose
    surname merely appeared INSIDE a longer word, which `find` can never match."""
    inj, _ = canary.generate(1, seed=14)
    surname = inj[0].surname
    assert canary.collisions(inj, [f"prefix{surname}suffix and more text"]) == {}
    assert "surname" in canary.collisions(inj, [f"the patient {surname} came in"])


def test_collisions_detects_telephone_across_punctuation():
    """The detector strips non-digits before matching, so the check must too --
    otherwise a corpus that formats the same number differently slips past."""
    inj, _ = canary.generate(30, seed=15)
    c = next(x for x in inj if x.tel)
    assert "tel" in canary.collisions([c], [f"call ({c.tel}) now"])
    assert "tel" in canary.collisions([c], ["call " + c.tel.replace("-", " ")])
