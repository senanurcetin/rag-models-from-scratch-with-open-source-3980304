import os
import sys
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d_02_prepare_content import is_unique_to_window, get_filtered_matches, merge_windows
from d_05_run_rag import format_context, clean_answer, build_prompt

# (id, sentence_number, content, file_name, distance)
def hit(sentence_number, file_name="a.txt"):
    return (sentence_number, sentence_number, f"s{sentence_number}", file_name, 0.1)


def test_same_file_within_window_is_not_unique():
    assert not is_unique_to_window([hit(10)], hit(20), group_window_size=5)
    assert is_unique_to_window([hit(10)], hit(21), group_window_size=5)


def test_other_file_is_always_unique():
    assert is_unique_to_window([hit(10)], hit(10, "b.txt"), group_window_size=5)


def test_filtered_matches_skips_overlaps_and_stops_at_limit():
    results = [hit(10), hit(12), hit(40), hit(11, "b.txt"), hit(80), hit(120), hit(200)]
    matches = get_filtered_matches(results, num_matches=4, group_window_size=5)
    assert [(m[3], m[1]) for m in matches] == [("a.txt", 10), ("a.txt", 40), ("b.txt", 11), ("a.txt", 80)]


def test_merge_windows_merges_touching_windows_per_file():
    merged = merge_windows(["a.txt", "a.txt", "b.txt", "a.txt"], [10, 21, 10, 50], group_window_size=5)
    assert merged == [("a.txt", 5, 26), ("a.txt", 45, 55), ("b.txt", 5, 15)]


def test_merge_windows_clamps_start_to_first_sentence():
    assert merge_windows(["a.txt"], [2], group_window_size=5) == [("a.txt", 1, 7)]


Row = namedtuple("Row", "id sentence_number content file_name")


def test_format_context_and_prompt():
    ctx = [[Row(1, 1, "First. ", "a.txt"), Row(2, 2, "Second.", "a.txt")], [Row(9, 1, "Other.", "b.txt")]]
    assert format_context(ctx) == "[Source: a.txt]\nFirst. Second.\n\n[Source: b.txt]\nOther."
    prompt = build_prompt(ctx, "Q?")
    assert prompt.startswith("<|content_start>\n") and "<|content_end>\n\nQ?" in prompt


def test_clean_answer_strips_think_block():
    assert clean_answer("<think>\nhmm\n</think>\n\nYes.") == "Yes."
    assert clean_answer("Yes.") == "Yes."
