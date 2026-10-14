//! Port of COLMAP's `src/colmap/util/cache.h`: [`LRUCache`], [`ThreadSafeLRUCache`] and
//! [`MemoryConstrainedLRUCache`]. Port of colmap-sharp's `Util/Cache.cs`. The SIFT CPU matcher
//! reads its per-image descriptor indexes through a [`ThreadSafeLRUCache`] (it implements
//! [`crate::feature::index::FeatureDescriptorIndexCache`]); COLMAP's MVS workspace caches
//! images and depth maps in a [`MemoryConstrainedLRUCache`]. Tests: `tests/util/cache.rs`
//! (`cache_test.cc` 1:1).
//!
//! Tier A (exact): plain bookkeeping.
//!
//! Translation notes:
//! - `std::shared_ptr<value_t>` is `Arc<V>` (`V` may be unsized, e.g. a trait object). The load
//!   function returns `Result<Arc<V>>`: a C++ load function that throws is an `Err`, which
//!   `Get` passes on with the cache left as it was (`LRUCache.ExceptionSafety`).
//! - `std::list` + `NodeHashMap<key, list iterator>` is a slab-backed doubly linked list
//!   (`LruList`) + a `HashMap<key, slot>`. Nothing iterates the map, so its random order
//!   cannot reach an output.
//! - The constructors' `THROW_CHECK_GT(max, 0)` is an `Err`; `THROW_CHECK_NOTNULL(load_fn)` has
//!   no counterpart (a Rust closure is never null), nor has `ThreadSafeLRUCache::Get`'s
//!   `THROW_CHECK_NOTNULL` of the loaded value (an `Arc` is never null).
//! - `ThreadSafeLRUCache`'s `shared_mutex` is a `Mutex` and its per-entry promise /
//!   `shared_future` is an `Entry` (a mutex-guarded slot and a condvar): the first caller
//!   for a key loads it outside the cache lock, later callers for the same key block until the
//!   same value (or the same error) is set. On `wasm32-unknown-unknown` (one thread) a waiter
//!   never finds a load in flight, so the condvar is never waited on.
//! - `MemoryConstrainedLRUCache`'s duck-typed `value->NumBytes()` is the [`NumBytes`] trait.
//!   `MaxNumElems` is declared in `cache.h` but never defined, so it has no port. Its `Get`
//!   re-looks-up the key after loading in case the load function inserted it re-entrantly;
//!   in Rust the load function cannot reach the cache it is stored in, so that branch has no
//!   counterpart. `Pop`'s and `UpdateNumBytes`'s underflow checks guard an internal invariant
//!   (the total is the sum of the stored sizes) and are assertions.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::check_gt;
use crate::util::check::{ColmapError, ErrorKind, Result};

/// The load function of [`LRUCache`] and [`MemoryConstrainedLRUCache`] (`LoadFn`).
pub type LoadFn<K, V> = Box<dyn FnMut(&K) -> Result<Arc<V>> + Send>;

/// The load function of [`ThreadSafeLRUCache`]: called outside the cache lock, possibly from
/// several threads at once (for different keys).
pub type SharedLoadFn<K, V> = Box<dyn Fn(&K) -> Result<Arc<V>> + Send + Sync>;

const NIL: usize = usize::MAX;

/// One list node: the key, its value and the neighbours toward the front (`prev`, more
/// recently used) and the back (`next`).
struct Node<K, T> {
    key: K,
    value: T,
    prev: usize,
    next: usize,
}

/// `std::list<std::pair<key, value>>` + `NodeHashMap<key, iterator>`: front is the most
/// recently used element. Slots of removed nodes are reused.
struct LruList<K, T> {
    nodes: Vec<Option<Node<K, T>>>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
    map: HashMap<K, usize>,
}

impl<K: Hash + Eq + Clone, T> LruList<K, T> {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            map: HashMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn node(&self, slot: usize) -> &Node<K, T> {
        self.nodes[slot].as_ref().expect("live LRU slot")
    }

    fn node_mut(&mut self, slot: usize) -> &mut Node<K, T> {
        self.nodes[slot].as_mut().expect("live LRU slot")
    }

    fn unlink(&mut self, slot: usize) {
        let (prev, next) = {
            let node = self.node(slot);
            (node.prev, node.next)
        };
        if prev == NIL {
            self.head = next;
        } else {
            self.node_mut(prev).next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.node_mut(next).prev = prev;
        }
    }

    fn link_front(&mut self, slot: usize) {
        let old_head = self.head;
        {
            let node = self.node_mut(slot);
            node.prev = NIL;
            node.next = old_head;
        }
        if old_head == NIL {
            self.tail = slot;
        } else {
            self.node_mut(old_head).prev = slot;
        }
        self.head = slot;
    }

    /// `emplace_front` + `map.emplace`; the key must not be present.
    fn push_front(&mut self, key: K, value: T) -> usize {
        let node = Node {
            key: key.clone(),
            value,
            prev: NIL,
            next: NIL,
        };
        let slot = match self.free.pop() {
            Some(slot) => {
                self.nodes[slot] = Some(node);
                slot
            }
            None => {
                self.nodes.push(Some(node));
                self.nodes.len() - 1
            }
        };
        self.link_front(slot);
        self.map.insert(key, slot);
        slot
    }

    /// `splice(begin, list, it)`: the key's node moves to the front. Returns its slot.
    fn touch(&mut self, key: &K) -> Option<usize> {
        let slot = *self.map.get(key)?;
        self.unlink(slot);
        self.link_front(slot);
        Some(slot)
    }

    fn remove_slot(&mut self, slot: usize) -> Node<K, T> {
        self.unlink(slot);
        let node = self.nodes[slot].take().expect("live LRU slot");
        self.free.push(slot);
        self.map.remove(&node.key);
        node
    }

    fn remove(&mut self, key: &K) -> Option<Node<K, T>> {
        let slot = *self.map.get(key)?;
        Some(self.remove_slot(slot))
    }

    fn pop_back(&mut self) -> Option<Node<K, T>> {
        (self.tail != NIL).then(|| self.remove_slot(self.tail))
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.free.clear();
        self.map.clear();
        self.head = NIL;
        self.tail = NIL;
    }
}

/// Port of `colmap::LRUCache`: least recently used cache. Whenever the cache size is exceeded,
/// the least recently used (by `get`) element is deleted.
pub struct LRUCache<K, V: ?Sized> {
    max_num_elems: usize,
    elems: LruList<K, Arc<V>>,
    load_fn: LoadFn<K, V>,
}

impl<K: Hash + Eq + Clone, V: ?Sized> LRUCache<K, V> {
    /// A cache of at most `max_num_elems` values computed by `load_fn`; `Err` when
    /// `max_num_elems` is 0.
    pub fn new(
        max_num_elems: usize,
        load_fn: impl FnMut(&K) -> Result<Arc<V>> + Send + 'static,
    ) -> Result<Self> {
        check_gt!(max_num_elems, 0);
        Ok(Self {
            max_num_elems,
            elems: LruList::new(),
            load_fn: Box::new(load_fn),
        })
    }

    /// `NumElems`: the number of elements in the cache.
    pub fn num_elems(&self) -> usize {
        self.elems.len()
    }

    /// `MaxNumElems`.
    pub fn max_num_elems(&self) -> usize {
        self.max_num_elems
    }

    /// `Exists`: whether the element with the given key is in the cache.
    pub fn exists(&self, key: &K) -> bool {
        self.elems.map.contains_key(key)
    }

    /// `Get`: the value of an element, either from the cache or newly computed. A load error is
    /// returned with the cache unchanged.
    pub fn get(&mut self, key: &K) -> Result<Arc<V>> {
        if let Some(slot) = self.elems.touch(key) {
            return Ok(self.elems.node(slot).value.clone());
        }
        // Call load_fn before modifying the cache data structures so that if it fails, the
        // cache remains in a consistent state.
        let value = (self.load_fn)(key)?;
        self.elems.push_front(key.clone(), value.clone());
        if self.elems.len() > self.max_num_elems {
            self.pop();
        }
        Ok(value)
    }

    /// `Evict`: manually evicts an element; true if it was in the cache.
    pub fn evict(&mut self, key: &K) -> bool {
        self.elems.remove(key).is_some()
    }

    /// `Pop`: removes the least recently used element (nothing when empty).
    pub fn pop(&mut self) {
        self.elems.pop_back();
    }

    /// `Clear`: removes all elements.
    pub fn clear(&mut self) {
        self.elems.clear();
    }
}

/// `ThreadSafeLRUCache::Entry`: the promise / `shared_future` of one key's value.
struct Entry<V: ?Sized> {
    state: Mutex<EntryState<V>>,
    ready: Condvar,
}

struct EntryState<V: ?Sized> {
    is_loading: bool,
    result: Option<Result<Arc<V>>>,
}

impl<V: ?Sized> Entry<V> {
    fn new() -> Self {
        Self {
            state: Mutex::new(EntryState {
                is_loading: false,
                result: None,
            }),
            ready: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, EntryState<V>> {
        // A panicking load function poisons nothing that matters here: the state is only a
        // flag and an optional result.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `promise.set_value` / `set_exception`.
    fn set(&self, result: Result<Arc<V>>) {
        self.lock().result = Some(result);
        self.ready.notify_all();
    }

    /// `shared_future.get()`: blocks until the value (or error) is set.
    fn wait(&self) -> Result<Arc<V>> {
        let mut state = self.lock();
        loop {
            if let Some(result) = &state.result {
                return result.clone();
            }
            state = self.ready.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Port of `colmap::ThreadSafeLRUCache`: an [`LRUCache`] safe to share between threads, where
/// concurrent requests for one key load it only once.
pub struct ThreadSafeLRUCache<K, V: ?Sized> {
    load_fn: SharedLoadFn<K, V>,
    cache: Mutex<LRUCache<K, Entry<V>>>,
}

impl<K: Hash + Eq + Clone + Send + 'static, V: ?Sized + Send + Sync + 'static>
    ThreadSafeLRUCache<K, V>
{
    /// A cache of at most `max_num_elems` values computed by `load_fn`; `Err` when
    /// `max_num_elems` is 0.
    pub fn new(
        max_num_elems: usize,
        load_fn: impl Fn(&K) -> Result<Arc<V>> + Send + Sync + 'static,
    ) -> Result<Self> {
        let cache = LRUCache::new(max_num_elems, |_: &K| Ok(Arc::new(Entry::new())))?;
        Ok(Self {
            load_fn: Box::new(load_fn),
            cache: Mutex::new(cache),
        })
    }

    fn lock(&self) -> MutexGuard<'_, LRUCache<K, Entry<V>>> {
        self.cache.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `NumElems`.
    pub fn num_elems(&self) -> usize {
        self.lock().num_elems()
    }

    /// `MaxNumElems`.
    pub fn max_num_elems(&self) -> usize {
        self.lock().max_num_elems()
    }

    /// `Exists`.
    pub fn exists(&self, key: &K) -> bool {
        self.lock().exists(key)
    }

    /// `Get`: the value from the cache, or loaded by this caller (outside the lock) when no
    /// other caller is loading it already; otherwise waits for that load. A failed load is
    /// evicted and its error returned to every caller that waited on it.
    pub fn get(&self, key: &K) -> Result<Arc<V>> {
        let (entry, should_load) = {
            let mut cache = self.lock();
            let entry = cache.get(key)?;
            let mut state = entry.lock();
            let should_load = !state.is_loading;
            state.is_loading = true;
            drop(state);
            (entry, should_load)
        };

        if should_load {
            match (self.load_fn)(key) {
                Ok(value) => entry.set(Ok(value)),
                Err(error) => {
                    // Evict the cache entry after load failed and set the error.
                    self.lock().evict(key);
                    entry.set(Err(error));
                }
            }
        }

        entry.wait()
    }

    /// `Evict`.
    pub fn evict(&self, key: &K) -> bool {
        self.lock().evict(key)
    }

    /// `Pop`.
    pub fn pop(&self) {
        self.lock().pop();
    }

    /// `Clear`.
    pub fn clear(&self) {
        self.lock().clear();
    }
}

/// `value->NumBytes()`, what [`MemoryConstrainedLRUCache`] charges for each element.
pub trait NumBytes {
    /// The element's size in memory.
    fn num_bytes(&self) -> usize;
}

/// Port of `colmap::MemoryConstrainedLRUCache`: least recently used cache constrained by the
/// memory its elements report. Whenever the limit is exceeded, least recently used elements
/// are deleted (but never the last one).
pub struct MemoryConstrainedLRUCache<K, V: ?Sized> {
    max_num_bytes: usize,
    num_bytes: usize,
    /// Each element with the byte count it was charged.
    elems: LruList<K, (Arc<V>, usize)>,
    load_fn: LoadFn<K, V>,
}

impl<K: Hash + Eq + Clone, V: ?Sized + NumBytes> MemoryConstrainedLRUCache<K, V> {
    /// A cache of at most `max_num_bytes` bytes of values computed by `load_fn`; `Err` when
    /// `max_num_bytes` is 0.
    pub fn new(
        max_num_bytes: usize,
        load_fn: impl FnMut(&K) -> Result<Arc<V>> + Send + 'static,
    ) -> Result<Self> {
        check_gt!(max_num_bytes, 0);
        Ok(Self {
            max_num_bytes,
            num_bytes: 0,
            elems: LruList::new(),
            load_fn: Box::new(load_fn),
        })
    }

    /// `NumBytes`: the bytes charged for the elements in the cache.
    pub fn num_bytes(&self) -> usize {
        self.num_bytes
    }

    /// `MaxNumBytes`.
    pub fn max_num_bytes(&self) -> usize {
        self.max_num_bytes
    }

    /// `NumElems`.
    pub fn num_elems(&self) -> usize {
        self.elems.len()
    }

    /// `Exists`.
    pub fn exists(&self, key: &K) -> bool {
        self.elems.map.contains_key(key)
    }

    fn shrink_to_limit(&mut self) {
        while self.num_bytes > self.max_num_bytes && self.elems.len() > 1 {
            self.pop();
        }
    }

    /// `Get`: the value from the cache, or loaded and charged its `num_bytes()`, after which
    /// least recently used elements are dropped while over the limit.
    pub fn get(&mut self, key: &K) -> Result<Arc<V>> {
        if let Some(slot) = self.elems.touch(key) {
            return Ok(self.elems.node(slot).value.0.clone());
        }
        let value = (self.load_fn)(key)?;
        let num_bytes = value.num_bytes();
        self.elems
            .push_front(key.clone(), (value.clone(), num_bytes));
        self.num_bytes += num_bytes;
        self.shrink_to_limit();
        Ok(value)
    }

    /// `Evict`.
    pub fn evict(&mut self, key: &K) -> bool {
        match self.elems.remove(key) {
            Some(node) => {
                self.num_bytes -= node.value.1;
                true
            }
            None => false,
        }
    }

    /// `Pop`.
    pub fn pop(&mut self) {
        if let Some(node) = self.elems.pop_back() {
            assert!(
                self.num_bytes >= node.value.1,
                "Unsigned underflow in MemoryConstrainedLRUCache::Pop"
            );
            self.num_bytes -= node.value.1;
        }
    }

    /// `UpdateNumBytes`: re-measures the element (now most recently used) after it changed
    /// size, then drops least recently used elements while over the limit. `Err`
    /// (`std::out_of_range`, from `unordered_map::at`) when the key is not in the cache.
    pub fn update_num_bytes(&mut self, key: &K) -> Result<()> {
        let Some(slot) = self.elems.touch(key) else {
            return Err(ColmapError::new(
                ErrorKind::OutOfRange,
                "MemoryConstrainedLRUCache::UpdateNumBytes: key not in the cache",
            ));
        };
        let node = self.elems.node_mut(slot);
        assert!(
            self.num_bytes >= node.value.1,
            "Unsigned underflow in MemoryConstrainedLRUCache::UpdateNumBytes"
        );
        self.num_bytes -= node.value.1;
        node.value.1 = node.value.0.num_bytes();
        self.num_bytes += node.value.1;
        self.shrink_to_limit();
        Ok(())
    }

    /// `Clear`.
    pub fn clear(&mut self) {
        self.elems.clear();
        self.num_bytes = 0;
    }
}
