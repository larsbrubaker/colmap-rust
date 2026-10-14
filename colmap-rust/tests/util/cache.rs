// Port of COLMAP's src/colmap/util/cache_test.cc, 1:1 (Suite.Name -> suite_name), testing
// colmap_rust::util::cache. Tier A (exact). `SizedElem::num_bytes` is an atomic so a test can
// change an element's size through the shared `Arc`, as the C++ test does through its
// `shared_ptr`. ConcurrentGet's gmock loader (each key loaded exactly once) is a per-key call
// counter checked after the threads join.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use colmap_rust::util::cache::{LRUCache, MemoryConstrainedLRUCache, NumBytes, ThreadSafeLRUCache};
use colmap_rust::util::check::{ColmapError, ErrorKind};

fn int_cache(max: usize) -> LRUCache<i32, i32> {
    LRUCache::new(max, |key: &i32| Ok(Arc::new(*key))).unwrap()
}

fn ts_cache(max: usize) -> ThreadSafeLRUCache<i32, i32> {
    ThreadSafeLRUCache::new(max, |key: &i32| Ok(Arc::new(*key))).unwrap()
}

struct SizedElem {
    num_bytes: AtomicUsize,
}

impl SizedElem {
    fn new(num_bytes: usize) -> Self {
        Self {
            num_bytes: AtomicUsize::new(num_bytes),
        }
    }

    fn set(&self, num_bytes: usize) {
        self.num_bytes.store(num_bytes, Ordering::SeqCst);
    }
}

impl NumBytes for SizedElem {
    fn num_bytes(&self) -> usize {
        self.num_bytes.load(Ordering::SeqCst)
    }
}

fn sized_cache(max: usize) -> MemoryConstrainedLRUCache<i32, SizedElem> {
    MemoryConstrainedLRUCache::new(max, |key: &i32| Ok(Arc::new(SizedElem::new(*key as usize))))
        .unwrap()
}

#[test]
fn lru_cache_empty() {
    let cache = int_cache(5);
    assert_eq!(cache.num_elems(), 0);
    assert_eq!(cache.max_num_elems(), 5);
}

/// The first five gets of every LRU test, then `Get(5)` evicting 0.
fn fill_lru(cache: &mut LRUCache<i32, i32>) {
    assert_eq!(cache.num_elems(), 0);
    for i in 0..5 {
        assert_eq!(*cache.get(&i).unwrap(), i);
        assert_eq!(cache.num_elems(), (i + 1) as usize);
        assert!(cache.exists(&i));
    }
}

#[test]
fn lru_cache_get() {
    let mut cache = int_cache(5);
    fill_lru(&mut cache);

    assert_eq!(*cache.get(&5).unwrap(), 5);
    assert_eq!(cache.num_elems(), 5);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&5));

    assert_eq!(*cache.get(&5).unwrap(), 5);
    assert_eq!(cache.num_elems(), 5);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&5));

    assert_eq!(*cache.get(&6).unwrap(), 6);
    assert_eq!(cache.num_elems(), 5);
    assert!(!cache.exists(&0));
    assert!(!cache.exists(&1));
    assert!(cache.exists(&6));
}

#[test]
fn lru_cache_evict() {
    let mut cache = int_cache(5);
    fill_lru(&mut cache);

    assert_eq!(*cache.get(&5).unwrap(), 5);
    assert_eq!(cache.num_elems(), 5);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&5));

    assert!(!cache.evict(&0));

    assert!(cache.evict(&1));
    assert_eq!(cache.num_elems(), 4);
    assert!(!cache.exists(&1));
}

#[test]
fn lru_cache_pop() {
    let mut cache = int_cache(5);
    fill_lru(&mut cache);

    assert_eq!(*cache.get(&5).unwrap(), 5);
    assert_eq!(cache.num_elems(), 5);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&5));

    for expected in [4, 3, 2, 1, 0, 0] {
        cache.pop();
        assert_eq!(cache.num_elems(), expected);
    }
}

#[test]
fn lru_cache_clear() {
    let mut cache = int_cache(5);
    fill_lru(&mut cache);

    cache.clear();
    assert_eq!(cache.num_elems(), 0);

    assert_eq!(*cache.get(&0).unwrap(), 0);
    assert_eq!(cache.num_elems(), 1);
    assert!(cache.exists(&0));
}

#[test]
fn memory_constrained_lru_cache_empty() {
    let cache = sized_cache(5);
    assert_eq!(cache.num_elems(), 0);
    assert_eq!(cache.num_bytes(), 0);
    assert_eq!(cache.max_num_bytes(), 5);
}

/// The first five gets of the memory-constrained tests (sizes 0..4, 10 bytes in total).
fn fill_sized(cache: &mut MemoryConstrainedLRUCache<i32, SizedElem>) {
    assert_eq!(cache.num_elems(), 0);
    for i in 0..5 {
        assert_eq!(cache.get(&i).unwrap().num_bytes(), i as usize);
        assert_eq!(cache.num_elems(), (i + 1) as usize);
        assert!(cache.exists(&i));
    }
}

#[test]
fn memory_constrained_lru_cache_get() {
    let mut cache = sized_cache(10);
    fill_sized(&mut cache);

    assert_eq!(cache.get(&5).unwrap().num_bytes(), 5);
    assert_eq!(cache.num_elems(), 2);
    assert_eq!(cache.num_bytes(), 9);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&5));

    assert_eq!(cache.get(&5).unwrap().num_bytes(), 5);
    assert_eq!(cache.num_elems(), 2);
    assert_eq!(cache.num_bytes(), 9);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&5));

    assert_eq!(cache.get(&6).unwrap().num_bytes(), 6);
    assert_eq!(cache.num_elems(), 1);
    assert_eq!(cache.num_bytes(), 6);
    assert!(!cache.exists(&0));
    assert!(!cache.exists(&1));
    assert!(cache.exists(&6));

    assert_eq!(cache.get(&1).unwrap().num_bytes(), 1);
    assert_eq!(cache.num_elems(), 2);
    assert_eq!(cache.num_bytes(), 7);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&1));
    assert!(cache.exists(&6));
}

#[test]
fn memory_constrained_lru_cache_pop() {
    let mut cache = sized_cache(10);
    fill_sized(&mut cache);

    assert_eq!(cache.get(&5).unwrap().num_bytes(), 5);
    assert_eq!(cache.num_elems(), 2);
    assert_eq!(cache.num_bytes(), 9);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&4));
    assert!(cache.exists(&5));

    cache.pop();
    assert_eq!(cache.num_elems(), 1);
    assert_eq!(cache.num_bytes(), 5);
    assert!(!cache.exists(&4));
    assert!(cache.exists(&5));

    cache.pop();
    assert_eq!(cache.num_elems(), 0);
    assert_eq!(cache.num_bytes(), 0);
}

#[test]
fn memory_constrained_lru_cache_evict() {
    let mut cache = sized_cache(10);
    fill_sized(&mut cache);

    assert_eq!(cache.get(&5).unwrap().num_bytes(), 5);
    assert_eq!(cache.num_elems(), 2);
    assert_eq!(cache.num_bytes(), 9);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&4));
    assert!(cache.exists(&5));

    assert!(!cache.evict(&0));

    assert!(cache.evict(&5));
    assert_eq!(cache.num_elems(), 1);
    assert_eq!(cache.num_bytes(), 4);
    assert!(cache.exists(&4));
    assert!(!cache.exists(&5));
}

#[test]
fn memory_constrained_lru_cache_clear() {
    let mut cache = sized_cache(10);
    fill_sized(&mut cache);

    cache.clear();
    assert_eq!(cache.num_elems(), 0);
    assert_eq!(cache.num_bytes(), 0);

    assert_eq!(cache.get(&1).unwrap().num_bytes(), 1);
    assert_eq!(cache.num_bytes(), 1);
    assert_eq!(cache.num_elems(), 1);
    assert!(cache.exists(&1));
}

#[test]
fn memory_constrained_lru_cache_update_num_bytes() {
    let mut cache = sized_cache(50);
    fill_sized(&mut cache);

    assert_eq!(cache.num_bytes(), 10);

    cache.get(&4).unwrap().set(3);
    assert_eq!(cache.num_bytes(), 10);
    cache.update_num_bytes(&4).unwrap();
    assert_eq!(cache.num_bytes(), 9);

    cache.get(&2).unwrap().set(3);
    assert_eq!(cache.num_bytes(), 9);
    cache.update_num_bytes(&2).unwrap();
    assert_eq!(cache.num_bytes(), 10);

    cache.get(&0).unwrap().set(40);
    assert_eq!(cache.num_bytes(), 10);
    cache.update_num_bytes(&0).unwrap();
    assert_eq!(cache.num_bytes(), 50);

    cache.clear();
    assert_eq!(cache.num_bytes(), 0);
    assert_eq!(cache.get(&2).unwrap().num_bytes(), 2);
    assert_eq!(cache.num_bytes(), 2);
}

#[test]
fn thread_safe_lru_cache_empty() {
    let cache = ts_cache(5);
    assert_eq!(cache.num_elems(), 0);
    assert_eq!(cache.max_num_elems(), 5);
}

fn fill_ts(cache: &ThreadSafeLRUCache<i32, i32>) {
    assert_eq!(cache.num_elems(), 0);
    for i in 0..5 {
        assert_eq!(*cache.get(&i).unwrap(), i);
        assert_eq!(cache.num_elems(), (i + 1) as usize);
        assert!(cache.exists(&i));
    }
}

#[test]
fn thread_safe_lru_cache_get() {
    let cache = ts_cache(5);
    fill_ts(&cache);

    assert_eq!(*cache.get(&5).unwrap(), 5);
    assert_eq!(cache.num_elems(), 5);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&5));

    assert_eq!(*cache.get(&5).unwrap(), 5);
    assert_eq!(cache.num_elems(), 5);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&5));

    assert_eq!(*cache.get(&6).unwrap(), 6);
    assert_eq!(cache.num_elems(), 5);
    assert!(!cache.exists(&0));
    assert!(!cache.exists(&1));
    assert!(cache.exists(&6));
}

#[test]
fn thread_safe_lru_cache_concurrent_get() {
    let signal = Arc::new((Mutex::new(false), Condvar::new()));
    let load_counts = Arc::new(Mutex::new(HashMap::<i32, usize>::new()));

    let loader_signal = signal.clone();
    let loader_counts = load_counts.clone();
    let cache = Arc::new(
        ThreadSafeLRUCache::new(2, move |key: &i32| {
            *loader_counts.lock().unwrap().entry(*key).or_default() += 1;
            match *key {
                0 => {
                    let (mutex, cv) = &*loader_signal;
                    *mutex.lock().unwrap() = true;
                    cv.notify_one();
                    Ok(Arc::new(2))
                }
                1 => Ok(Arc::new(4)),
                _ => panic!("unexpected Load({key})"),
            }
        })
        .unwrap(),
    );

    let cache1 = cache.clone();
    let thread1 = thread::spawn(move || assert_eq!(*cache1.get(&0).unwrap(), 2));
    let cache2 = cache.clone();
    let signal2 = signal.clone();
    let thread2 = thread::spawn(move || {
        let (mutex, cv) = &*signal2;
        let loaded = cv
            .wait_while(mutex.lock().unwrap(), |loaded| !*loaded)
            .unwrap();
        assert!(*loaded);
        assert_eq!(*cache2.get(&0).unwrap(), 2);
    });

    thread1.join().unwrap();
    thread2.join().unwrap();

    assert_eq!(*cache.get(&1).unwrap(), 4);

    let counts = load_counts.lock().unwrap();
    assert_eq!(counts.get(&0), Some(&1));
    assert_eq!(counts.get(&1), Some(&1));
}

#[test]
fn thread_safe_lru_cache_evict() {
    let cache = ts_cache(5);
    fill_ts(&cache);

    assert_eq!(*cache.get(&5).unwrap(), 5);
    assert_eq!(cache.num_elems(), 5);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&5));

    assert!(!cache.evict(&0));

    assert!(cache.evict(&1));
    assert_eq!(cache.num_elems(), 4);
    assert!(!cache.exists(&1));
}

#[test]
fn thread_safe_lru_cache_pop() {
    let cache = ts_cache(5);
    fill_ts(&cache);

    assert_eq!(*cache.get(&5).unwrap(), 5);
    assert_eq!(cache.num_elems(), 5);
    assert!(!cache.exists(&0));
    assert!(cache.exists(&5));

    for expected in [4, 3, 2, 1, 0, 0] {
        cache.pop();
        assert_eq!(cache.num_elems(), expected);
    }
}

#[test]
fn thread_safe_lru_cache_clear() {
    let cache = ts_cache(5);
    fill_ts(&cache);

    cache.clear();
    assert_eq!(cache.num_elems(), 0);

    assert_eq!(*cache.get(&0).unwrap(), 0);
    assert_eq!(cache.num_elems(), 1);
    assert!(cache.exists(&0));
}

#[test]
fn lru_cache_exception_safety() {
    // If load_fn_ throws, the cache should remain in a consistent state.
    let load_count = Arc::new(AtomicUsize::new(0));
    let counter = load_count.clone();
    let mut cache = LRUCache::new(5, move |key: &i32| {
        counter.fetch_add(1, Ordering::SeqCst);
        if *key == 3 {
            return Err(ColmapError::new(ErrorKind::RuntimeError, "load failed"));
        }
        Ok(Arc::new(*key * 10))
    })
    .unwrap();

    // Load some initial entries.
    assert_eq!(*cache.get(&0).unwrap(), 0);
    assert_eq!(*cache.get(&1).unwrap(), 10);
    assert_eq!(*cache.get(&2).unwrap(), 20);
    assert_eq!(cache.num_elems(), 3);

    // Attempting to load key=3 should throw.
    let error = cache.get(&3).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RuntimeError);

    // The cache should still be in a valid state with the original 3 entries.
    assert_eq!(cache.num_elems(), 3);
    assert!(cache.exists(&0));
    assert!(cache.exists(&1));
    assert!(cache.exists(&2));
    assert!(!cache.exists(&3));

    // Existing entries should still be retrievable.
    assert_eq!(*cache.get(&0).unwrap(), 0);
    assert_eq!(*cache.get(&1).unwrap(), 10);
    assert_eq!(*cache.get(&2).unwrap(), 20);

    // Loading a new valid key should work fine.
    assert_eq!(*cache.get(&4).unwrap(), 40);
    assert_eq!(cache.num_elems(), 4);
    assert!(cache.exists(&4));
    assert_eq!(load_count.load(Ordering::SeqCst), 5);
}

#[test]
fn memory_constrained_lru_cache_pop_num_bytes_consistency() {
    // Pop should maintain correct num_bytes tracking.
    let mut cache = sized_cache(100);

    cache.get(&10).unwrap();
    cache.get(&20).unwrap();
    cache.get(&5).unwrap();
    assert_eq!(cache.num_bytes(), 35);
    assert_eq!(cache.num_elems(), 3);

    // Pop should remove the LRU element (key=10, 10 bytes) and update num_bytes.
    cache.pop();
    assert_eq!(cache.num_bytes(), 25);
    assert_eq!(cache.num_elems(), 2);

    cache.pop();
    assert_eq!(cache.num_bytes(), 5);
    assert_eq!(cache.num_elems(), 1);

    cache.pop();
    assert_eq!(cache.num_bytes(), 0);
    assert_eq!(cache.num_elems(), 0);

    // Pop on empty cache should be safe.
    cache.pop();
    assert_eq!(cache.num_bytes(), 0);
    assert_eq!(cache.num_elems(), 0);
}

#[test]
fn memory_constrained_lru_cache_update_num_bytes_consistency() {
    // Should correctly track byte counts even when elements shrink.
    let mut cache = sized_cache(100);

    cache.get(&10).unwrap();
    cache.get(&20).unwrap();
    assert_eq!(cache.num_bytes(), 30);

    // Shrink element from 20 bytes to 5 bytes.
    cache.get(&20).unwrap().set(5);
    cache.update_num_bytes(&20).unwrap();
    assert_eq!(cache.num_bytes(), 15);

    // Shrink element to 0 bytes.
    cache.get(&10).unwrap().set(0);
    cache.update_num_bytes(&10).unwrap();
    assert_eq!(cache.num_bytes(), 5);

    // Grow it back.
    cache.get(&10).unwrap().set(8);
    cache.update_num_bytes(&10).unwrap();
    assert_eq!(cache.num_bytes(), 13);
}
