"""
Eviction controller -- WHEN to evict, and HOW MUCH.

eviction.py answers WHICH (an ordering of keys, coldest first). This file owns
the other two questions, and is the only place that actually deletes anything:

    WHEN        usage_ratio >= watermark?                    should_evict()
    HOW MUCH    free at least ratio * capacity bytes          bytes_to_free()
    (act)       walk the policy's ranking, delete from L1     evict()

The worked example from eviction.py, with the numbers the tests use:

    capacity 1000, watermark 0.9, ratio 0.2, usage 950 after a put
    should_evict()   950 / 1000 = 0.95 >= 0.9           -> True
    bytes_to_free()  0.2 * 1000                         -> 200
    evict()          coldest first: [k3, k7, k1, k9]
                     delete k3 (80 bytes),  120 to go
                     delete k7 (150 bytes), budget met  -> stop, return [k3, k7]

Real LMCache runs this check on a background thread at ~1 Hz. Here there is
one thread and the check runs right after a put -- the only moment usage can
go up. Same decision, no thread.

Stage 5 note: when L2 arrives, "delete" in evict() becomes "write the buffer to
disk, then delete". Only this file changes for that; eviction.py does not.
"""

from l1 import L1Cache, buffer_bytes
from eviction import LRUPolicy


class EvictionController:
    """
    Keeps L1 under its watermark by deleting the coldest entries.

    Holds a reference to L1 (to read usage and delete) and to the policy (to
    get the ranking). Neither of those holds a reference back.
    """

    def __init__(
        self,
        l1: L1Cache,
        policy: LRUPolicy,
        watermark: float = 0.9,
        ratio: float = 0.2,
    ):
        if not 0 < watermark <= 1:
            raise ValueError(f"watermark must be in (0, 1], got {watermark}")
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        self.l1 = l1
        self.policy = policy
        self.watermark = watermark
        self.ratio = ratio

        # Running totals for run.py's summary.
        self.num_evictions = 0
        self.bytes_evicted = 0

    # -- WHEN ------------------------------------------------------------------

    def should_evict(self) -> bool:
        """
        True when L1's usage ratio has reached the watermark.

        Decide: >= or >? The design doc says "usage >= watermark". The tests
        assume >=; if you choose > instead, change the test and write down why.
        """
        return self.l1.usage_ratio >= self.watermark
        
    # -- HOW MUCH --------------------------------------------------------------

    def bytes_to_free(self) -> int:
        """
        The eviction budget: ratio * L1 capacity, in bytes.

        Note what it is a fraction OF. Not of current usage, not of the gap to
        the watermark -- of capacity. Work out what each alternative would do
        when usage is barely over the watermark.
        """
        return int(self.ratio*self.l1.capacity_bytes)

    # -- act ---------------------------------------------------------------------

    def evict(self) -> list[int]:
        """
        Delete entries, coldest first, until bytes_to_free() bytes are freed or
        there is nothing left to delete. Returns the evicted keys, in order.

        Unconditional: it does not check should_evict(). maybe_evict() does.

        Things to get right (each one has a test):

        1.  Sizes come from L1, not the policy -- the policy never sees a byte.
            L1 has no size_of(key), but it has get(), and l1.py has the
            module-level buffer_bytes().
            Read the size BEFORE you delete.

        2.  A key the policy ranks but L1 no longer holds must not be credited
            as freed bytes. (That is exactly the stale-key bug described in
            LRUPolicy.forget's docstring.)

        3.  Every evicted key has to leave the policy too, or coldest() will
            keep nominating it. Who calls policy.forget()? Careful:
            LRUPolicy.forget's docstring says "not called by the eviction path"
            and LRUPolicy.coldest's says "each delete leads to a forget()".
            They disagree. Decide which is right, fix the wrong docstring.

        4.  Stop when the budget is met -- not one entry later, not one early.
            And stop if the ranking runs out before the budget is met.

        5.  Update num_evictions and bytes_evicted.
        """
        budget = self.bytes_to_free()
        freed = 0
        evicted : list[int] = []
        for key in self.policy.coldest():
            buffer = self.l1.get(key)
            # Leaves the ranking either way: an evicted key is gone, and a
            # stale one would otherwise be nominated again on every pass.
            self.policy.forget(key)

            if buffer is None:
                # Stale: the policy ranked it, L1 no longer holds it. Nothing
                # was freed, so nothing is credited.
                continue

            size = buffer_bytes(buffer)  # before the delete, while it exists
            self.l1.delete(key)

            freed += size
            evicted.append(key)
            self.num_evictions += 1
            self.bytes_evicted += size

            if freed >= budget:
                break

        return evicted

    def maybe_evict(self) -> list[int]:
        """
        The call cache.py makes after every successful put: evict if over the
        watermark, otherwise do nothing. Returns the evicted keys (often []).
        """
        if self.should_evict():
            return self.evict()
        else:
            return []

    def __repr__(self) -> str:
        return (
            f"EvictionController(watermark {self.watermark}, ratio {self.ratio}, "
            f"{self.num_evictions} evicted, {self.bytes_evicted} bytes)"
        )
