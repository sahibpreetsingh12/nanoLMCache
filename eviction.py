"""
Eviction policy -- WHICH entry leaves when the cache is full.


WHY THIS FILE DID NOT EXIST UNTIL NOW
-------------------------------------
Today l1.put() returns False when an entry does not fit, and that is the end of
it. A cache that refuses never has to choose a victim, so "hot" and "cold" have
had no meaning in this project so far. They acquire one at exactly one moment:
the cache is full, the incoming entry is worth keeping, so something already in
there has to go. This file answers "which one", and nothing else.


eviction.py VS controller.py
----------------------------
Three separate decisions hide inside the word "eviction":

    WHEN        has usage crossed the watermark?            controller.py
    HOW MUCH    free this many bytes, then stop             controller.py
    WHICH       of the keys I hold, this is the coldest     eviction.py  <-- here

Walk one eviction with real numbers. L1 capacity 1000 bytes, watermark 0.9,
ratio 0.2, and a put has just pushed usage to 950:

    controller   950/1000 = 0.95, that is over 0.9             -> act      WHEN
    controller   0.2 * 1000 = free at least 200 bytes          -> budget   HOW MUCH
    eviction     "coldest first: [k3, k7, k1, k9]"             -> ranking  WHICH
    controller   l1.delete(k3), that was 80 bytes, 120 to go
    controller   l1.delete(k7), that was 150 bytes, budget met -> stop

Read that list of steps again and notice what each side never touched:

    eviction.py never saw a byte, never deleted anything, and was never told
    how much to free or why it was being asked. It returned an ordering.

    controller.py never had an opinion about which key deserved to survive. It
    had a budget and a list, and it walked the list until the budget was met.

One line: controller.py has the budget, eviction.py has the opinion.

They are separate files because they change for unrelated reasons. Swap LRU for
random and the watermark logic is untouched. Move the watermark from 0.9 to 0.7
and the ranking is untouched. Real LMCache draws the same line -- a background
thread polls at roughly 1Hz (WHEN), and the policy it calls is one of LRU,
IsolatedLRU or noop (WHICH).

The deeper reason is Stage 5. When L2 arrives, eviction stops meaning "discard"
and starts meaning "write the buffer to disk, then drop it from memory". The
action changes completely. The ranking does not change at all -- the coldest
entry is still the right one to demote. Everything that has to be rewritten for
L2 is in controller.py; this file will not be touched.

A consequence worth stating: this class never imports l1, never holds a
reference to a cache, and never deletes anything. Keys go in, keys come out.
Which also means it is testable with no pool, no buffers and no cache at all --
feed it a sequence of touches, ask what order it wants to give things up in.


WHY A DICT IS IN THE PICTURE, AND WHY IT IS NOT ENOUGH
------------------------------------------------------
This is not an analogy. It is the thing you reach for first, and the reason it
fails is the reason this class exists.

1.  L1's store already IS a dict, and dicts preserve insertion order since 3.7.
    l1.keys() is documented as "keys in insertion order -- oldest first". So the
    first instinct is that LRU is already here for free: evict l1.keys()[0].

2.  It is not, because insertion order answers the wrong question. A dict
    records when a key was first ADDED. LRU needs to know when it was last
    USED. Those are the same number only if nothing is ever read twice, which
    in a cache is precisely backwards.

        d = {}
        d["a"] = 1
        d["b"] = 2
        d["a"] = 99        # assigning to an existing key does NOT move it
        list(d)            # ['a', 'b']  -- "a" is still first

    So a dict gives you FIFO: oldest-inserted leaves first, no matter that it
    was read on every single request. An entry written once and read forever is
    the hottest thing in the cache, and FIFO evicts it first.

3.  You can force the move by hand -- del d[k] then d[k] = v puts it at the
    back. But now recording a READ means writing to L1's store, and l1.py is
    explicit that L1 does not know what a hit is ("hit and miss are properties
    of a lookup, and lookups belong to the orchestrator"). Order of use is a
    different fact from the buffer itself, and it should not have to touch the
    buffer to get recorded.

4.  So: a second structure, holding keys only, ordered by use instead of by
    insertion. collections.OrderedDict is a dict that also exposes
    move_to_end(), which is step 3 done properly and in O(1). That one method
    is the entire reason to prefer it over a plain dict here.

5.  Its values are never read. This is an ordered SET of keys, and
    OrderedDict[int, None] is the cheapest way to say that with the standard
    library.


WHAT MAKES AN ENTRY "RECENT"
----------------------------
Two events, and both must call touch():

    a put    the entry is brand new, so it is the most recently used thing
    a hit    l1.get() returned a buffer -- somebody wanted this

The second is the one that is easy to forget, and forgetting it leaves you with
point 2 above: a working FIFO wearing an LRU's name. Both calls belong in
cache.py, not l1.py -- the orchestrator is the only layer that knows a lookup
hit.
"""

from collections import OrderedDict


class LRUPolicy:
    """
    Least-recently-used ranking over cache keys. Most recent at the back.

    The whole class in one trace:

        p = LRUPolicy()
        p.touch(100)       # [100]
        p.touch(200)       # [100, 200]
        p.touch(300)       # [100, 200, 300]
        p.touch(100)       # [200, 300, 100]   <- 100 was reused, so it moves
        p.coldest()        # [200, 300, 100]   -> 200 is the one to drop
        p.forget(300)      # [200, 100]        <- 300 left L1 some other way

    Line 4 is the only interesting line in the file. A dict would have left the
    order at [100, 200, 300] and nominated 100 -- the entry that had just been
    used -- as the victim.

    touch() and forget() are O(1). coldest() is O(n) in the number of entries.
    """

    def __init__(self) -> None:
        # TODO: self._order: OrderedDict[int, None] = OrderedDict()
        #       Keys only. Front = coldest, back = most recently used.
        raise NotImplementedError

    def touch(self, key: int) -> None:
        """
        Record that `key` was just used. It becomes the most recent.

        Called from cache.py in two places: after l1.put() returns True, and
        after l1.get() returns a buffer. Those are the only two events that
        make an entry recent.

        One body has to cover both cases:
            key is new        -> add it at the back
            key already here  -> MOVE it to the back

        The second case is the one a plain dict gets wrong, so it is the one to
        write deliberately. Two lines: insert, then move_to_end.
        """
        raise NotImplementedError

    def forget(self, key: int) -> None:
        """
        Stop ranking `key`. It is no longer in L1.

        Not called by the eviction path -- the controller already knows it
        dropped that key. This is for every OTHER way an entry can leave:
        someone calls l1.delete() directly, or l1.clear() wipes the store.

        Skip it and the policy slowly fills with keys that are not in L1 any
        more. Then coldest() nominates one, the controller calls l1.delete(),
        gets False, and credits itself with bytes it never freed -- so the
        eviction loop stops while usage is still above the watermark.

        Silent on a key it does not hold. The caller is telling us something is
        gone; it being gone already is not an error. pop with a default.
        """
        raise NotImplementedError

    def coldest(self) -> list[int]:
        """
        Every key, coldest first. The controller takes as many as it needs.

        Two choices in that sentence worth defending.

        Why the whole ranking and not coldest(n): the policy has no idea what
        an entry costs. Bytes live in L1, and the controller's budget is "200
        bytes", not "2 entries". So this returns the only thing it can know --
        the order -- and the controller walks it, asking L1 for each size, until
        the budget is met.

        Why a list and not an iterator over the OrderedDict: the controller
        deletes as it walks, each delete leads to a forget(), and forget()
        mutates the very OrderedDict being iterated. That is a RuntimeError. A
        snapshot makes the caller safe without the caller having to know, and a
        list of ints costs nothing next to the buffers being moved.

        TODO: list(self._order) -- front is already coldest.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        # TODO: entry count, plus the coldest key or two -- the numbers you
        #       want in front of you when an eviction test fails.
        raise NotImplementedError
