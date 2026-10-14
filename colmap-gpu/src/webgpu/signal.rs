//! A one-shot future completed from a wgpu callback (`Queue::on_submitted_work_done`,
//! `Buffer::map_async`). On the desktop the callback has already run when the device poll
//! returns, so the first poll is ready; in the browser it runs from the JS event loop and wakes
//! the awaiting task. No async runtime: the waker is whatever the caller's executor supplies.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

struct State<T> {
    value: Option<T>,
    waker: Option<Waker>,
}

/// The receiving half: resolves to the value the callback sends.
pub(crate) struct Signal<T> {
    state: Arc<Mutex<State<T>>>,
}

impl<T: Send + 'static> Signal<T> {
    /// A signal and the callback that completes it.
    pub(crate) fn new() -> (Self, impl FnOnce(T) + Send + 'static) {
        let state = Arc::new(Mutex::new(State {
            value: None,
            waker: None,
        }));
        let sender = {
            let state = Arc::clone(&state);
            move |value: T| {
                // A poisoned lock means the awaiting side panicked; nothing is left to wake.
                if let Ok(mut state) = state.lock() {
                    state.value = Some(value);
                    if let Some(waker) = state.waker.take() {
                        waker.wake();
                    }
                }
            }
        };
        (Self { state }, sender)
    }
}

impl<T> Future for Signal<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        match state.value.take() {
            Some(value) => Poll::Ready(value),
            None => {
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}
