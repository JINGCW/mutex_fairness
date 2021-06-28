#ifndef _MUTEX_HPP_
#define _MUTEX_HPP_
#include <atomic>
#include <chrono>
#include <windows.h>
#include <thread>
#include <iostream>

#define NAMESPACE_BEGIN(name) namespace name {
#define NAMESPACE_END(name) }
NAMESPACE_BEGIN(XQ)
using time_point_t = std::chrono::system_clock::time_point;
using clock_t = std::chrono::system_clock;

enum class mutex_state : uint16_t {
  MutexLocked = 0x1,
  MutexWoken = 0x10,
  MutexStarving = 0x100,
  MutexWaiterShift = 3,
  // Mutex fairness.
  //
  // Mutex can be in 2 modes of operations: normal and starvation.
  // In normal mode waiters are queued in FIFO order, but a woken up waiter
  // does not own the mutex and competes with new arriving goroutines over
  // the ownership. New arriving goroutines have an advantage -- they are
  // already running on CPU and there can be lots of them, so a woken up
  // waiter has good chances of losing. In such case it is queued at front
  // of the wait queue. If a waiter fails to acquire the mutex for more than 1ms,
  // it switches mutex to the starvation mode.
  //
  // In starvation mode ownership of the mutex is directly handed off from
  // the unlocking goroutine to the waiter at the front of the queue.
  // New arriving goroutines don't try to acquire the mutex even if it appears
  // to be unlocked, and don't try to spin. Instead they queue themselves at
  // the tail of the wait queue.
  //
  // If a waiter receives ownership of the mutex and sees that either
  // (1) it is the last waiter in the queue, or (2) it waited for less than 1 ms,
  // it switches mutex back to normal operation mode.
  //
  // Normal mode has considerably better performance as a goroutine can acquire
  // a mutex several times in a row even if there are blocked waiters.
  // Starvation mode is important to prevent pathological cases of tail latency.
};
static const int64_t StarvationThresholdNs = 1e6;

#define XQ_CAST(type, value) static_cast<type>(value)
#define S1_CAST XQ_CAST(uint16_t, s1)
#define S2_CAST XQ_CAST(uint16_t, s2)

#define OPERATOR_FUNC(op, type1, type2) \
uint16_t operator op(type1 s1,type2 s2) \
{return S1_CAST op S2_CAST;}

OPERATOR_FUNC(|, mutex_state, mutex_state)
OPERATOR_FUNC(==, int, mutex_state)
OPERATOR_FUNC(&, int, mutex_state)

#define STATE(name) mutex_state::name
#define LOCKED STATE(MutexLocked)
#define WOKEN STATE(MutexWoken)
#define STARVE STATE(MutexStarving)
#define SHIFT STATE(MutexWaiterShift)

#ifdef __SSE2__
#include <emmintrin.h>
inline void spin_loop_pause() noexcept { _mm_pause(); }
#elif defined(_MSC_VER) && _MSC_VER >= 1800 && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
inline void spin_loop_pause() noexcept {
  _mm_pause();
}
#else
inline void spin_loop_pause() noexcept {}
#endif

inline time_point_t runtime_nanotime() { return clock_t::now(); }

#define RUNTIME_COUNT(start) \
std::chrono::duration_cast<std::chrono::nanoseconds>(runtime_nanotime() - (start)).count()

typedef struct mutex {
  void lock() {
    lock_slow();
  }

  void lock_slow() {
    time_point_t wait_start_time;
    bool wait_start_time_flag = false;
    bool starving = false;
    bool awoke = false;
    size_t iter = 0;
    int state_old = state.load();
    for (;;) {
      // Don't spin in starvation mode, ownership is handed off to waiters
      // so we won't be able to acquire the mutex anyway.
      if ((state_old & (LOCKED | STARVE)) == LOCKED) {
        // Active spinning makes sense.
        // Try to set mutexWoken flag to inform Unlock
        // to not wake other blocked goroutines.
        if (!awoke && ((state_old & WOKEN) == 0) && state_old >> SHIFT != 0 &&
            state.compare_exchange_strong(state_old, state_old | WOKEN)) {
          awoke = true;
        }
        spin_loop_pause();
        iter++;
        state_old = state;
        continue;
      }
      auto state_new = state_old;
      // Don't try to acquire starving mutex, state_new arriving goroutines must queue.
      if ((state_old & STARVE) == 0)
        state_new |= LOCKED;

      if ((state_old & (LOCKED | STARVE)) != 0)
        state_new += 1 << SHIFT;
      // The current goroutine switches mutex to starvation mode.
      // But if the mutex is currently unlocked, don't do the switch.
      // Unlock expects that starving mutex has waiters, which will not
      // be true in this case.
      if (starving && (state_old & LOCKED) != 0)
        state_new |= STARVE;

      if (awoke) {
        // The goroutine has been woken from sleep,
        // so we need to reset the flag in either case.
        if ((state_new & WOKEN) == 0) {
          throw ("sync: inconsistent mutex state");
        }
//            state_new &^= WOKEN;
        auto temp = state_new ^ WOKEN;
        state_new &= temp;
      }

      if (state.compare_exchange_weak(state_old, state_new)) {
        if ((state_old & (LOCKED | STARVE)) == 0) {
          break;// locked the mutex with CAS
        }
        // If we were already waiting before, queue at the front of the queue.
        auto queueLifo = wait_start_time_flag != 0;
        if (wait_start_time_flag == 0) {
          wait_start_time = runtime_nanotime();
        }
        int queueLifo_int = queueLifo ? 1 : 0;
        sema.compare_exchange_weak(queueLifo_int, 1);
        starving = starving|| (RUNTIME_COUNT(wait_start_time)> StarvationThresholdNs);
        state_old = state.load();
        if ((state_old & STARVE) != 0) {
          // If this goroutine was woken and mutex is in starvation mode,
          // ownership was handed off to us but mutex is in somewhat
          // inconsistent state: mutexLocked is not set and we are still
          // accounted as waiter. Fix that.
          if ((state_old & (LOCKED | WOKEN)) != 0
              || state_old >> SHIFT == 0) {
            throw ("sync: inconsistent mutex state");
          }
          auto delta = int((LOCKED - 1) << SHIFT);
          if (!starving || state_old >> SHIFT == 1) {
            // Exit starvation mode.
            // Critical to do it here and consider wait time.
            // Starvation mode is so inefficient, that two goroutines
            // can go lock-step infinitely once they switch mutex
            // to starvation mode.
            delta -= STARVE;
          }
          state.fetch_add(delta);
          break;
        }
        awoke = true;
        iter = 0;
      } else {
        state_old = state;
      }
    }
  }

  void Unlock() {
    // Fast path: drop lock bit.
    auto state_new = state.fetch_add(-LOCKED);
    if (state_new != 0) {
      // Outlined slow path to allow inlining the fast path.
      // To hide unlockSlow during tracing we skip one extra frame when tracing GoUnblock.
      UnlockSlow(state_new);
    }
  }
  void UnlockSlow(int NewState) {
    if (((NewState + LOCKED) & LOCKED) == 0) {
      throw ("sync: unlock of unlocked mutex");
    }
    if ((NewState & STARVE) == 0) {
      auto state_old = NewState;
      for (;;) {
        // If there are no waiters or a goroutine has already
        // been woken or grabbed the lock, no need to wake anyone.
        // In starvation mode ownership is directly handed off from unlocking
        // goroutine to the next waiter. We are not part of this chain,
        // since we did not observe mutexStarving when we unlocked the mutex above.
        // So get off the way.
        if (state_old >> SHIFT == 0
            || (state_old & (LOCKED | WOKEN | STARVE)) != 0) {
          return;
        }
        // Grab the right to wake someone.
        NewState = ((state_old - 1) << SHIFT) | WOKEN;
        if (state.compare_exchange_weak(state_old, NewState)) {
          int tmp = 0;
          sema.compare_exchange_weak(tmp, 1);
          return;
        }
        state_old = state;
      }
    } else {
      // Starving mode: handoff mutex ownership to the next waiter, and yield
      // our time slice so that the next waiter can start to run immediately.
      // Note: mutexLocked is not set, the waiter will set it after wakeup.
      // But mutex is still considered locked if mutexStarving is set,
      // so NewState coming goroutines won't acquire it.
      int tmp = 1;
      sema.compare_exchange_weak(tmp, 1);
    }
  }

  std::atomic<int> state;
  std::atomic<int> sema;
} mutex;
NAMESPACE_END(XQ)
#endif //_MUTEX_HPP_
