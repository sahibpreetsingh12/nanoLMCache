"""
Eviction policy — which entry goes when something has to.

Nothing in this project has needed the words "hot" or "cold" until now, and it
is worth being precise about why. Today l1.put() returns False when an entry
does not fit, and that is the end of it: the chunk gets prefilled again next
time, which is slower but correct. A cache that refuses is never asked to
choose. Recency only becomes a *question* at one specific moment -- the cache is
full, the incoming entry is worth keeping, and therefore something already in
there has to leave. This file exists to answer that question and nothing else.

Three ideas get bundled together in the word "eviction", and keeping them in
separate files is most of the design:

    WHEN       usage_ratio has crossed a watermark          controller.py
    HOW MUCH   free `ratio` worth of bytes, then stop       controller.py
    WHICH      of the entries I hold, this one is coldest    <- here

They are independent. You can keep LRU and change the watermark from 0.9 to
0.7; you can keep both and swap LRU for random. Real LMCache makes the same
cut: a background thread polls at roughly 1Hz (that is WHEN), and the policy
it calls is one of LRU, IsolatedLRU, or noop (that is WHICH).

A policy is a RANKING, not an action
------------------------------------
Note what is missing from the interface below: this class never imports L1,
never holds a reference to one, and never deletes anything. It is handed keys
and it hands back keys. The eviction *loop* -- call l1.delete, subtract the
bytes, check whether that was enough -- lives in controller.py.

That split is not tidiness, it is Stage 5. When L2 arrives, eviction stops
meaning "discard" and starts meaning "write the buffer to disk, then drop it
from memory". The action changes completely. The ranking does not change at
all: the coldest entry is still the right one to demote. If this class did the
deleting, Stage 5 would have to reach inside it and teach it about disk.

It also means the policy is testable with no pool, no buffers and no cache --
feed it a sequence of touches, ask what order it wants to give things up in.

What "recently used" has to mean here
-------------------------------------
Two events make an entry recent, and both must call touch():

    a put    the entry is brand new, so it is the most recently used thing
    a hit    l1.get() returned a buffer -- somebody wanted this

The second is the one that is easy to miss. An entry that is never written
again but is read on every request is the hottest thing in the cache, and a
policy that only hears about puts cannot tell it apart from a dead entry.
Wiring l1.get() to touch the policy on a hit is a cache.py change, and it is
the change that turns this from FIFO into LRU.

GOTCHA: a plain dict is FIFO, not LRU
-------------------------------------
Python dicts preserve insertion order, which is why l1.keys() looks like it is
already most of a policy. It is not. Assigning to a key that already exists
keeps its ORIGINAL position:

    d = {}; d["a"] = 1; d["b"] = 2
    d["a"] = 99          # "a" is still first, not last
    list(d)              # ['a', 'b']

So re-touching an entry through a dict is a no-op on the ordering, and the
oldest-inserted entry gets evicted no matter how often it was used. Use
collections.OrderedDict and its move_to_end(), which exists precisely for
this; the alternative is `del d[k]` followed by reinsertion, which is the same
thing written by hand.

The values are never read -- this is an ordered *set* of keys. OrderedDict[int,
None] is the cheapest way to say that in the standard library.
"""

from collections import OrderedDict


class LRUPolicy:
    """
    Least-recently-used ranking over cache keys.

    Most recent at the back, coldest at the front. Every operation is O(1)
    except coldest(), which is O(n) in the number of entries.
    """

    def __init__(self) -> None:
        # TODO: one OrderedDict[int, None], used as an ordered set of keys.
        raise NotImplementedError

    def touch(self, key: int) -> None:
        """
        Mark a key as the most recently used. Called on a put and on a hit.

        Must handle both the new key and the already-present key with the same
        line of reasoning: either way the key ends up at the back. The
        already-present case is the one a plain dict gets wrong.
        """
        # TODO: insert, then move_to_end.
        raise NotImplementedError

    def forget(self, key: int) -> None:
        """
        Drop a key the policy should stop ranking.

        Needed because entries can leave L1 by paths that are not eviction --
        l1.delete() called directly, or l1.clear(). A policy that did not hear
        about those would eventually nominate a victim that is no longer there,
        and l1.delete() would return False while the controller believed it had
        freed those bytes. The eviction loop would then stop early, under the
        watermark it was trying to get below.

        Silent on a key it does not hold: the caller is telling us something is
        gone, and it being gone twice is not an error.
        """
        # TODO: pop with a default.
        raise NotImplementedError

    def coldest(self) -> list[int]:
        """
        Every key, coldest first. The controller takes as many as it needs.

        Why the whole ranking rather than coldest(n): the policy does not know
        what an entry costs. Bytes live in L1, and "free 10% of capacity" is a
        number of bytes, not a number of entries. So the policy answers the
        only question it can -- the order -- and the controller walks that list
        asking L1 for each size until it has freed enough.

        Why a list rather than an iterator over the OrderedDict: the controller
        deletes as it walks, and each delete calls forget(), which mutates the
        same OrderedDict being iterated. That raises RuntimeError. Returning a
        snapshot makes the caller safe by construction, and a list of ints is
        nothing next to the buffers being moved.
        """
        # TODO: list(self._order) -- front is coldest.
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
