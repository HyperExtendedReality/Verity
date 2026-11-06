//! Bounded CPU-side storage for immutable geometry pages.
//!
//! This module does not allocate GPU resources or synchronize with a renderer.
//! The byte budget covers resident page payloads, not cache metadata, memory
//! owned by the source, or the temporary page returned by a source while a load
//! is in progress. A source must return the exact size it advertised.

use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;

/// A zero-based page index in a [`PageSource`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PageId(pub u32);

/// Supplies immutable page payloads on demand.
///
/// The page count and the size and contents of each page must remain stable
/// for the lifetime of the cache. Implementations may read from files, memory,
/// or another backing store. Loading does not imply GPU residency.
pub trait PageSource {
    /// Error returned when querying or loading a page fails.
    type Error: Error + 'static;

    /// Number of valid pages, indexed from zero.
    fn page_count(&self) -> u32;
    /// Exact payload size for a valid page.
    fn page_size(&mut self, id: PageId) -> Result<usize, Self::Error>;
    /// Read a valid page into immutable owned storage.
    fn load_page(&mut self, id: PageId) -> Result<Box<[u8]>, Self::Error>;
}

/// An operation on a CPU page cache failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum PageError<E> {
    /// The page index is outside the source's page count.
    InvalidPage(PageId),
    /// The page cannot fit even in an empty cache.
    PageTooLarge {
        id: PageId,
        size: usize,
        budget: usize,
    },
    /// Pinned pages leave insufficient space for the requested page.
    InsufficientSpace { id: PageId, size: usize },
    /// An operation requires the page to be resident.
    NotResident(PageId),
    /// A pinned page cannot be unloaded.
    Pinned(PageId),
    /// An unpin requires a matching pin.
    NotPinned(PageId),
    /// The page's pin count cannot be incremented.
    PinOverflow(PageId),
    /// The source returned a payload of a different size than advertised.
    SizeMismatch {
        id: PageId,
        expected: usize,
        actual: usize,
    },
    /// A page-size query or load failed in the backing source.
    Source(E),
}

impl<E> fmt::Display for PageError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPage(id) => write!(f, "invalid page {}", id.0),
            Self::PageTooLarge { id, size, budget } => {
                write!(f, "page {} size {size} exceeds budget {budget}", id.0)
            }
            Self::InsufficientSpace { id, size } => write!(
                f,
                "insufficient unpinned space for page {} of size {size}",
                id.0
            ),
            Self::NotResident(id) => write!(f, "page {} is not resident", id.0),
            Self::Pinned(id) => write!(f, "page {} is pinned", id.0),
            Self::NotPinned(id) => write!(f, "page {} is not pinned", id.0),
            Self::PinOverflow(id) => write!(f, "page {} pin count overflow", id.0),
            Self::SizeMismatch {
                id,
                expected,
                actual,
            } => write!(f, "page {} has size {actual}, expected {expected}", id.0),
            Self::Source(_) => write!(f, "failed to access page source"),
        }
    }
}

impl<E: Error + 'static> Error for PageError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            _ => None,
        }
    }
}

struct ResidentPage {
    bytes: Box<[u8]>,
    pins: usize,
}

/// A byte-budgeted, least-recently-used CPU page cache.
///
/// A successful [`load`](Self::load), [`get`](Self::get), or
/// [`pin`](Self::pin) marks a resident page as most recently used. Eviction
/// scans from least recently used to most recently used, skipping pinned pages.
/// Each pin must be balanced with an unpin; pin coarse fallback pages for as
/// long as they must remain available. Explicit unloading also respects pins.
/// Failed loads leave existing resident pages and their LRU order unchanged.
pub struct PageCache<S: PageSource> {
    source: S,
    page_count: u32,
    budget: usize,
    used: usize,
    pages: BTreeMap<PageId, ResidentPage>,
    lru: VecDeque<PageId>,
}

impl<S: PageSource> PageCache<S> {
    /// Create an empty cache with the specified maximum resident payload bytes.
    pub fn new(source: S, budget: usize) -> Self {
        let page_count = source.page_count();
        Self {
            source,
            page_count,
            budget,
            used: 0,
            pages: BTreeMap::new(),
            lru: VecDeque::new(),
        }
    }

    /// Maximum number of resident payload bytes.
    pub fn budget(&self) -> usize {
        self.budget
    }
    /// Number of resident payload bytes.
    pub fn resident_bytes(&self) -> usize {
        self.used
    }
    /// Number of currently resident pages, including empty pages.
    pub fn resident_pages(&self) -> usize {
        self.pages.len()
    }
    /// Number of pages exposed by the source when the cache was created.
    pub fn page_count(&self) -> u32 {
        self.page_count
    }

    fn validate(&self, id: PageId) -> Result<(), PageError<S::Error>> {
        if id.0 >= self.page_count {
            Err(PageError::InvalidPage(id))
        } else {
            Ok(())
        }
    }

    fn touch(&mut self, id: PageId) {
        if let Some(index) = self.lru.iter().position(|&entry| entry == id) {
            self.lru.remove(index);
        }
        self.lru.push_back(id);
    }

    /// Load a page if necessary, evicting least-recently-used unpinned pages.
    ///
    /// Source failures, size mismatches, and insufficient space do not evict
    /// anything. The source may temporarily allocate the incoming payload in
    /// addition to the currently resident bytes.
    pub fn load(&mut self, id: PageId) -> Result<(), PageError<S::Error>> {
        self.validate(id)?;
        if self.pages.contains_key(&id) {
            self.touch(id);
            return Ok(());
        }
        let size = self.source.page_size(id).map_err(PageError::Source)?;
        if size > self.budget {
            return Err(PageError::PageTooLarge {
                id,
                size,
                budget: self.budget,
            });
        }
        let mut needed = size.saturating_sub(self.budget - self.used);
        for candidate in &self.lru {
            let page = &self.pages[candidate];
            if page.pins == 0 {
                needed = needed.saturating_sub(page.bytes.len());
            }
        }
        if needed != 0 {
            return Err(PageError::InsufficientSpace { id, size });
        }
        let bytes = self.source.load_page(id).map_err(PageError::Source)?;
        if bytes.len() != size {
            return Err(PageError::SizeMismatch {
                id,
                expected: size,
                actual: bytes.len(),
            });
        }
        while size > self.budget - self.used {
            let index = self
                .lru
                .iter()
                .position(|candidate| self.pages[candidate].pins == 0)
                .expect("space for the page was checked before loading");
            let victim = self
                .lru
                .remove(index)
                .expect("LRU index came from the queue");
            let page = self
                .pages
                .remove(&victim)
                .expect("LRU entries are resident");
            self.used -= page.bytes.len();
        }
        self.used += size; // The budget check above proves this cannot overflow.
        self.pages.insert(id, ResidentPage { bytes, pins: 0 });
        self.touch(id);
        Ok(())
    }

    /// Return a resident page's immutable bytes, marking it recently used.
    /// Returns `None` for a valid page that is not currently resident.
    pub fn get(&mut self, id: PageId) -> Result<Option<&[u8]>, PageError<S::Error>> {
        self.validate(id)?;
        if !self.pages.contains_key(&id) {
            return Ok(None);
        }
        self.touch(id);
        Ok(self.pages.get(&id).map(|page| page.bytes.as_ref()))
    }

    /// Load and pin a page. Each successful call requires one matching unpin.
    pub fn pin(&mut self, id: PageId) -> Result<(), PageError<S::Error>> {
        self.validate(id)?;
        if self
            .pages
            .get(&id)
            .is_some_and(|page| page.pins == usize::MAX)
        {
            return Err(PageError::PinOverflow(id));
        }
        self.load(id)?;
        let page = self
            .pages
            .get_mut(&id)
            .expect("successful load makes page resident");
        page.pins += 1;
        Ok(())
    }

    /// Release one pin without unloading the page.
    pub fn unpin(&mut self, id: PageId) -> Result<(), PageError<S::Error>> {
        self.validate(id)?;
        let page = self.pages.get_mut(&id).ok_or(PageError::NotResident(id))?;
        if page.pins == 0 {
            return Err(PageError::NotPinned(id));
        }
        page.pins -= 1;
        Ok(())
    }

    /// Unload an unpinned page. Returns whether it was resident.
    pub fn unload(&mut self, id: PageId) -> Result<bool, PageError<S::Error>> {
        self.validate(id)?;
        let Some(page) = self.pages.get(&id) else {
            return Ok(false);
        };
        if page.pins != 0 {
            return Err(PageError::Pinned(id));
        }
        let size = page.bytes.len();
        self.pages.remove(&id);
        self.used -= size;
        if let Some(index) = self.lru.iter().position(|&entry| entry == id) {
            self.lru.remove(index);
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    struct Source {
        pages: Vec<Vec<u8>>,
        fail: Option<PageId>,
        wrong_size: Option<PageId>,
    }

    impl PageSource for Source {
        type Error = io::Error;
        fn page_count(&self) -> u32 {
            self.pages.len() as u32
        }
        fn page_size(&mut self, id: PageId) -> io::Result<usize> {
            Ok(self.pages[id.0 as usize].len())
        }
        fn load_page(&mut self, id: PageId) -> io::Result<Box<[u8]>> {
            if self.fail == Some(id) {
                return Err(io::Error::other("load failed"));
            }
            if self.wrong_size == Some(id) {
                return Ok(Box::new([]));
            }
            Ok(self.pages[id.0 as usize].clone().into_boxed_slice())
        }
    }

    fn source(pages: &[&[u8]]) -> Source {
        Source {
            pages: pages.iter().map(|page| page.to_vec()).collect(),
            fail: None,
            wrong_size: None,
        }
    }

    #[test]
    fn boundaries_and_actual_bytes() {
        let mut cache = PageCache::new(source(&[b"", b"ab", b"abc"]), 2);
        assert_eq!(cache.page_count(), 3);
        assert!(matches!(
            cache.load(PageId(3)),
            Err(PageError::InvalidPage(PageId(3)))
        ));
        assert!(matches!(
            cache.get(PageId(u32::MAX)),
            Err(PageError::InvalidPage(_))
        ));
        assert!(matches!(
            cache.load(PageId(2)),
            Err(PageError::PageTooLarge { .. })
        ));
        cache.load(PageId(0)).unwrap();
        cache.load(PageId(1)).unwrap();
        assert_eq!(cache.resident_bytes(), 2);
        assert_eq!(cache.resident_pages(), 2);
        assert_eq!(cache.get(PageId(0)).unwrap(), Some(&b""[..]));
        assert_eq!(cache.get(PageId(1)).unwrap(), Some(&b"ab"[..]));
        assert_eq!(cache.unload(PageId(1)).unwrap(), true);
        assert_eq!(cache.unload(PageId(1)).unwrap(), false);
        assert_eq!(cache.get(PageId(1)).unwrap(), None);
        assert_eq!(cache.resident_bytes(), 0);
        let mut zero = PageCache::new(source(&[b"", b"a"]), 0);
        zero.load(PageId(0)).unwrap();
        assert!(matches!(
            zero.load(PageId(1)),
            Err(PageError::PageTooLarge { .. })
        ));
    }

    #[test]
    fn eviction_skips_pinned_pages_and_can_remove_multiple_victims() {
        let mut cache = PageCache::new(source(&[b"a", b"bb", b"cc", b"ddd"]), 5);
        cache.pin(PageId(0)).unwrap();
        cache.load(PageId(1)).unwrap();
        cache.load(PageId(2)).unwrap();
        cache.load(PageId(3)).unwrap();
        assert_eq!(cache.get(PageId(0)).unwrap(), Some(&b"a"[..]));
        assert_eq!(cache.get(PageId(1)).unwrap(), None);
        assert_eq!(cache.get(PageId(2)).unwrap(), None);
        assert_eq!(cache.get(PageId(3)).unwrap(), Some(&b"ddd"[..]));
        assert_eq!(cache.resident_bytes(), 4);
    }

    #[test]
    fn lru_eviction_and_reload() {
        let mut cache = PageCache::new(source(&[b"aa", b"bb", b"cc"]), 4);
        cache.load(PageId(0)).unwrap();
        cache.load(PageId(1)).unwrap();
        cache.get(PageId(0)).unwrap();
        cache.load(PageId(2)).unwrap();
        assert_eq!(cache.get(PageId(1)).unwrap(), None);
        assert_eq!(cache.get(PageId(0)).unwrap(), Some(&b"aa"[..]));
        cache.load(PageId(1)).unwrap();
        assert_eq!(cache.get(PageId(2)).unwrap(), None);
        assert_eq!(cache.get(PageId(1)).unwrap(), Some(&b"bb"[..]));
        assert_eq!(cache.resident_bytes(), 4);
    }

    #[test]
    fn pins_protect_fallback_and_are_counted() {
        let mut cache = PageCache::new(source(&[b"aa", b"bb", b"ccc"]), 4);
        cache.pin(PageId(0)).unwrap();
        cache.pin(PageId(0)).unwrap();
        cache.load(PageId(1)).unwrap();
        assert!(matches!(cache.unload(PageId(0)), Err(PageError::Pinned(_))));
        assert!(matches!(
            cache.load(PageId(2)),
            Err(PageError::InsufficientSpace { .. })
        ));
        assert_eq!(cache.get(PageId(1)).unwrap(), Some(&b"bb"[..]));
        cache.unpin(PageId(0)).unwrap();
        assert!(matches!(cache.unload(PageId(0)), Err(PageError::Pinned(_))));
        cache.unpin(PageId(0)).unwrap();
        assert!(matches!(
            cache.unpin(PageId(0)),
            Err(PageError::NotPinned(_))
        ));
        cache.load(PageId(2)).unwrap();
        assert_eq!(cache.get(PageId(0)).unwrap(), None);
        assert_eq!(cache.get(PageId(2)).unwrap(), Some(&b"ccc"[..]));
        assert!(matches!(
            cache.unpin(PageId(0)),
            Err(PageError::NotResident(_))
        ));
    }

    #[test]
    fn failures_preserve_residents_and_order() {
        let mut backing = source(&[b"aa", b"bb", b"cc"]);
        backing.fail = Some(PageId(2));
        let mut cache = PageCache::new(backing, 4);
        cache.load(PageId(0)).unwrap();
        cache.load(PageId(1)).unwrap();
        assert!(matches!(cache.load(PageId(2)), Err(PageError::Source(_))));
        cache.source.fail = None;
        cache.source.wrong_size = Some(PageId(2));
        assert!(matches!(
            cache.load(PageId(2)),
            Err(PageError::SizeMismatch {
                expected: 2,
                actual: 0,
                ..
            })
        ));
        cache.source.wrong_size = None;
        cache.load(PageId(2)).unwrap();
        assert_eq!(cache.get(PageId(0)).unwrap(), None);
        assert_eq!(cache.get(PageId(1)).unwrap(), Some(&b"bb"[..]));
        assert_eq!(cache.resident_bytes(), 4);
    }

    #[test]
    fn pin_overflow_is_reported_without_changing_order() {
        let mut cache = PageCache::new(source(&[b"a", b"b"]), 2);
        cache.pin(PageId(0)).unwrap();
        cache.load(PageId(1)).unwrap();
        cache.pages.get_mut(&PageId(0)).unwrap().pins = usize::MAX;
        assert!(matches!(
            cache.pin(PageId(0)),
            Err(PageError::PinOverflow(_))
        ));
        assert_eq!(cache.lru.front(), Some(&PageId(0)));
    }
}
