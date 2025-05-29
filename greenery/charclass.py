# pylint: disable=fixme,too-many-locals,too-many-branches

from __future__ import annotations

__all__ = (
    "Charclass",
    "DIGIT",
    "DOT",
    "NONDIGITCHAR",
    "NONSPACECHAR",
    "NONWORDCHAR",
    "NULLCHARCLASS",
    "SPACECHAR",
    "WORDCHAR",
    "escapes",
    "negate",
    "shorthand",
    "repartition",
)

from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, Iterator, List, Mapping, Tuple

ALLOWED_INTERVALS = [(0x09, 0x0D), (0x20, 0x7E)]         # inclusive
ALLOWED_CHARS     = {(cp) for lo, hi in ALLOWED_INTERVALS
                            for cp in range(lo, hi + 1)}
ALPHABET_SIZE     = len(ALLOWED_CHARS)                   # 100

def _in_alphabet(cp: int) -> bool:
    return cp in ALLOWED_CHARS

def negate(ord_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Complement of *ord_ranges* **within ALLOWED_INTERVALS**.
    ord_ranges must be sorted & disjoint.
    """
    out: list[tuple[int, int]] = []
    rng_iter = iter(ord_ranges)
    try:
        lo, hi = next(rng_iter)
    except StopIteration:
        lo = hi = None

    for lo_a, hi_a in ALLOWED_INTERVALS:
        cursor = lo_a
        # consume every input range that intersects this allowed interval
        while lo is not None and lo <= hi_a:
            if hi < lo_a:                         # left of allowed block
                lo, hi = next(rng_iter, (None, None))
                continue
            if cursor < lo:                       # gap before this range
                out.append((cursor, lo - 1))
            cursor = hi + 1                       # skip the taken area
            lo, hi = next(rng_iter, (None, None))
        if cursor <= hi_a:                        # tail of allowed block
            out.append((cursor, hi_a))

    return out




def collapse_ord_ranges(ord_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Assume all existing ord ranges are sorted, and also disjoint
    So no cases of [[12, 17], [2, 3]] or [[4, 6], [7, 8]].
    """
    collapsed: List[Tuple[int, int]] = []

    for ord_range in sorted(ord_ranges):
        if not collapsed or collapsed[-1][1] + 1 < ord_range[0]:
            collapsed.append(ord_range)
        elif ord_range[1] > collapsed[-1][1]:
            # merge into previous
            collapsed[-1] = (collapsed[-1][0], ord_range[1])

    return collapsed


@dataclass(frozen=True, init=False)
class Charclass:
    """
    A `Charclass` is basically a `frozenset` of symbols.
    A `Charclass` with the `negated` flag set is assumed
    to contain every symbol that is in the alphabet of all symbols but not
    explicitly listed inside the frozenset. e.g. [^a]. This is very handy
    if the full alphabet is extremely large, but also requires dedicated
    combination functions.
    """

    ord_ranges: List[Tuple[int, int]]
    negated: bool

    def __init__(
        self, ranges: str | Tuple[Tuple[str, str], ...] = "", negated: bool = False
    ):
        if isinstance(ranges, str):
            ranges = tuple((char, char) for char in ranges)
        if not isinstance(ranges, tuple):
            raise TypeError(f"Bad ranges: {ranges!r}")
        for r in ranges:
            if len(r) != 2 or r[0] > r[1]:
                raise ValueError(f"Bad range: {r!r}")
            for char in r:
                if not isinstance(char, str):
                    raise TypeError(f"Can't put {char!r} in a `Charclass`")
                if len(char) != 1:
                    raise ValueError("`Charclass` can only contain single chars")

                # Ensure the char lies inside the allowed set.
                if not _in_alphabet(ord(char)):
                    raise ValueError(
                        f"{char!r} (U+{ord(char):04X}) is not printable ASCII."
                    )

        # Rebalance ranges!
        ord_ranges = [(ord(first), ord(last)) for first, last in ranges]
        ord_ranges = collapse_ord_ranges(ord_ranges)

        object.__setattr__(self, "ord_ranges", tuple(ord_ranges))
        object.__setattr__(self, "negated", negated)

    def __lt__(self, other: Charclass, /) -> bool:
        if self.negated < other.negated:
            return True
        if (
            self.negated == other.negated
            and self.ord_ranges[0][0] < other.ord_ranges[0][0]
        ):
            return True
        return False

    def __eq__(self, other: object, /) -> bool:
        return (
            isinstance(other, Charclass)
            and self.ord_ranges == other.ord_ranges
            and self.negated == other.negated
        )

    def __hash__(self, /) -> int:
        return hash((self.ord_ranges, self.negated))

    # These are the characters carrying special meanings when they appear
    # "outdoors" within a regular expression. To be interpreted literally, they
    # must be escaped with a backslash.
    allSpecial: ClassVar[frozenset[str]] = frozenset("\\[]|().?*+{}")

    # These are the characters carrying special meanings when they appear
    # INSIDE a character class (delimited by square brackets) within a regular
    # expression. To be interpreted literally, they must be escaped with a
    # backslash. Notice how much smaller this class is than the one above; note
    # also that the hyphen and caret do NOT appear above.
    classSpecial: ClassVar[frozenset[str]] = frozenset("\\[]^-")

    def __str__(self, /) -> str:
        # pylint: disable=too-many-return-statements

        # e.g. \w
        if self in shorthand:
            return shorthand[self]

        # e.g. [^a]
        if self.negated:
            return f"[^{self.escape()}]"

        # single character, not contained inside square brackets.
        if len(self.ord_ranges) == 1 and self.ord_ranges[0][0] == self.ord_ranges[0][1]:
            u = self.ord_ranges[0][0]
            char = chr(u)

            # e.g. if char is "\t", return "\\t"
            if char in escapes:
                return escapes[char]

            if char in Charclass.allSpecial:
                return f"\\{char}"

            # If char is an ASCII control character, don't print it directly,
            # return a hex escape sequence e.g. "\\x00". Note that this
            # includes tab and other characters already handled above
            if 0 <= u <= 0x1F or u == 0x7F:
                return f"\\x{u:02x}"

            return char

        # multiple characters (or possibly 0 characters)
        return f"[{self.escape()}]"

    # ── internal ----------------------------------------------------
    def _effective_ord_ranges(self) -> List[Tuple[int, int]]:
        """
        The intervals that are really in this class *right now*.
        If the class is negated we complement the stored ranges on demand.
        """
        return self.ord_ranges if not self.negated else negate(list(self.ord_ranges))

    def escape(self, /) -> str:
        def escape_char(char: str, /) -> str:
            if char in Charclass.classSpecial:
                return f"\\{char}"
            if char in escapes:
                return escapes[char]

            # If char is an ASCII control character, don't print it directly,
            # return a hex escape sequence e.g. "\\x00". Note that this
            # includes tab and other characters already handled above
            if 0 <= ord(char) <= 0x1F or ord(char) == 0x7F:
                return f"\\x{ord(char):02x}"

            return char

        output = ""

        for first_u, last_u in self.ord_ranges:
            # there's no point in putting a range when the whole thing is
            # 3 characters or fewer. "abc" -> "abc" but "abcd" -> "a-d"
            if last_u <= first_u + 2:
                # "a" or "ab" or "abc" or "abcd"
                for u in range(first_u, last_u + 1):
                    output += escape_char(chr(u))
            else:
                # "a-b" or "a-c" or "a-d"
                output += escape_char(chr(first_u)) + "-" + escape_char(chr(last_u))

        return output

    def __repr__(self, /) -> str:
        sign = "~" if self.negated else ""
        ranges = tuple(
            (chr(first_u), chr(last_u)) for (first_u, last_u) in self.ord_ranges
        )
        return f"{sign}Charclass({ranges!r})"

    def reduce(self, /) -> Charclass:
        # `Charclass`es cannot be reduced.
        return self

    def empty(self, /) -> bool:
        return not self.ord_ranges and not self.negated

    # set operations
    def negate(self, /) -> Charclass:
        """
        Negate the current `Charclass`. e.g. [ab] becomes [^ab]. Call
        using "charclass2 = ~charclass1"
        """
        ranges = tuple(
            (chr(first_u), chr(last_u)) for (first_u, last_u) in self.ord_ranges
        )
        return Charclass(ranges, negated=not self.negated)

    def __invert__(self, /) -> Charclass:
        return self.negate()

    # Charclass
    def get_chars(self, limit: int | None = None) -> Iterator[str]:
        """
        Yield all characters in the class.
        Use `limit` when you only want a prefix (e.g. for debugging).
        """
        seen = 0
        for first_u, last_u in self._effective_ord_ranges():
            for u in range(first_u, last_u + 1):
                if limit is not None and seen >= limit:
                    return
                yield chr(u)
                seen += 1


    # Charclass
    def num_chars(self) -> int:
        num = sum(hi - lo + 1 for lo, hi in self.ord_ranges)
        return num if not self.negated else ALPHABET_SIZE - num

    # Charclass
    def accepts(self, char: str) -> bool:
        # Check that char is in
        if not _in_alphabet(ord(char)):
            raise ValueError(
                f"{char!r} (U+{ord(char):04X}) is not printable ASCII."
            )

        u = ord(char)
        in_positive = any(lo <= u <= hi for lo, hi in self.ord_ranges)
        return in_positive ^ self.negated  # XOR does the trick

    def reversed(self, /) -> Charclass:
        return self

    def union(self, other: Charclass, /) -> Charclass:
        # TODO: make this able to efficiently unite many Charclasses at once,
        # again
        self_ord_ranges = list(self.ord_ranges)
        if self.negated:
            self_ord_ranges = negate(self_ord_ranges)

        other_ord_ranges = list(other.ord_ranges)
        if other.negated:
            other_ord_ranges = negate(other_ord_ranges)

        new_ord_ranges = []
        new_ord_ranges.extend(self_ord_ranges)
        new_ord_ranges.extend(other_ord_ranges)
        new_ord_ranges = collapse_ord_ranges(new_ord_ranges)

        new_negated = self.negated or other.negated
        if new_negated:
            new_ord_ranges = negate(new_ord_ranges)
        new_ranges = tuple(
            (chr(first_u), chr(last_u)) for (first_u, last_u) in new_ord_ranges
        )
        return Charclass(new_ranges, new_negated)

    __or__ = union

    def issubset(self, other: Charclass, /) -> bool:
        return self | other == other

    def intersection(self, other: Charclass, /) -> Charclass:
        # TODO: is this actually efficient?
        # TODO: make this able to efficiently intersect many Charclasses at once,
        # again
        return ~(~self | ~other)

    __and__ = intersection


# Standard character classes
WORDCHAR = Charclass("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz")
DIGIT = Charclass("0123456789")
SPACECHAR = Charclass("\t\n\v\f\r ")

# This `Charclass` expresses "no possibilities at all"
# and can never match anything.
NULLCHARCLASS = Charclass()

NONWORDCHAR = ~WORDCHAR
NONDIGITCHAR = ~DIGIT
NONSPACECHAR = ~SPACECHAR
DOT = ~NULLCHARCLASS

# Textual representations of standard character classes
shorthand: Mapping[Charclass, str] = {
    WORDCHAR: "\\w",
    DIGIT: "\\d",
    SPACECHAR: "\\s",
    NONWORDCHAR: "\\W",
    NONDIGITCHAR: "\\D",
    NONSPACECHAR: "\\S",
    DOT: ".",
}

# Characters which users may escape in a regex instead of inserting them
# literally. In ASCII order:
escapes: Mapping[str, str] = {
    "\t": "\\t",  # tab
    "\n": "\\n",  # line feed
    "\v": "\\v",  # vertical tab
    "\f": "\\f",  # form feed
    "\r": "\\r",  # carriage return
}


def repartition(
    charclasses: Iterable[Charclass],
) -> Mapping[Charclass, Iterable[Charclass]]:
    """
    Accept an iterable of `Charclass`es which may overlap somewhat.
    Construct a minimal collection of `Charclass`es which partition the space
    of all possible characters and can be combined to create all of the
    originals.
    Return a map from each original `Charclass` to its constituent pieces.
    """
    ord_range_boundaries = set()
    for charclass in charclasses:
        for first_u, last_u in charclass.ord_ranges:
            ord_range_boundaries.add(first_u)
            ord_range_boundaries.add(last_u + 1)
    ord_range_boundaries_2 = sorted(ord_range_boundaries)

    ord_ranges = []
    for i, ord_range_boundary in enumerate(ord_range_boundaries_2):
        if i + 1 < len(ord_range_boundaries_2):
            ord_ranges.append((ord_range_boundary, ord_range_boundaries_2[i + 1] - 1))

    # Group all of the possible ranges by "signature".
    # A signature is a tuple of Booleans telling us which character classes
    # a particular range is mentioned in.
    # (Whether it's *accepted* is actually not relevant.)
    signatures: Dict[Tuple[bool, ...], List[Tuple[int, int]]] = {}
    for ord_range in ord_ranges:
        signature = []
        for charclass in charclasses:
            ord_range_in_charclass = False
            for x in charclass.ord_ranges:
                if x[0] <= ord_range[0] and ord_range[1] <= x[1]:
                    ord_range_in_charclass = True
                    break
            signature.append(ord_range_in_charclass)
        signature2 = tuple(signature)
        if signature2 not in signatures:
            signatures[signature2] = []
        signatures[signature2].append(ord_range)

    # From the signatures we can gather the new Charclasses
    newcharclasses = []
    newcharclasses.append(
        ~Charclass(
            tuple((chr(first_u), chr(last_u)) for (first_u, last_u) in ord_ranges)
        )
    )
    for ord_ranges2 in signatures.values():
        newcharclasses.append(
            Charclass(
                tuple((chr(first_u), chr(last_u)) for (first_u, last_u) in ord_ranges2)
            )
        )

    # Now compute the breakdowns
    partition: Dict[Charclass, List[Charclass]] = {}
    for charclass in charclasses:
        partition[charclass] = []
        for newcharclass in newcharclasses:
            if newcharclass.issubset(charclass):
                partition[charclass].append(newcharclass)

    return partition
