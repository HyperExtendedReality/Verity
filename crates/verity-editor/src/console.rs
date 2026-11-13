//! Editor messages for the Console panel. A global so code without the
//! editor at hand, such as `Editor::error`, can log too.

use std::sync::Mutex;

/// Oldest messages drop past this.
const LIMIT: usize = 1000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    Info,
    Error,
}

static LOG: Mutex<Vec<(Level, String)>> = Mutex::new(Vec::new());

pub fn log(level: Level, message: impl Into<String>) {
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    if log.len() == LIMIT {
        log.remove(0);
    }
    log.push((level, message.into()));
}

pub fn info(message: impl Into<String>) {
    log(Level::Info, message);
}

pub fn error(message: impl Into<String>) {
    log(Level::Error, message);
}

/// Run `f` on the messages, oldest first; it may clear them.
pub fn with(f: impl FnOnce(&mut Vec<(Level, String)>)) {
    f(&mut LOG.lock().unwrap_or_else(|e| e.into_inner()));
}
