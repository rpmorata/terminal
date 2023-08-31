// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "atomic.h"

namespace til
{
    // ticket_lock implements a fair lock.
    // Unlike SRWLOCK, std::mutex, etc., forward progress for each thread is guaranteed.
    // It exploits the fact that WaitOnAddress/WakeByAddressSingle form a natural queue.
    // Compared to a SRWLOCK this implementation is significantly more unsafe to use:
    // Forgetting to call unlock or calling unlock more than once will lead to deadlocks.
    struct ticket_lock
    {
        void lock() noexcept
        {
            _state.fetch_add(1);

            for (uint32_t state = 1;;)
            {
                // In the first loop iteration we'll try to transition the lock from entirely unlocked (state = 0)
                // to the first one to having it locked (state = 1 | locked). This way, the first thread to arrive
                // here will never run into til::atomic_wait() unlike all other threads.
                if (_state.compare_exchange_strong(state, state | locked, std::memory_order_acquire))
                {
                    return;
                }

                do
                {
                    til::atomic_wait(_state, state);
                    state = _state.load(std::memory_order_relaxed);
                } while (state & locked);
            }
        }

        void unlock() noexcept
        {
            _state.fetch_sub(1 | locked, std::memory_order_release);
            // MSDN says about WakeByAddressSingle:
            // If multiple threads are waiting for this address, the system wakes the first thread to wait.
            til::atomic_notify_one(_state);
        }

    private:
        static constexpr uint32_t locked = 0x80000000;
        alignas(std::hardware_destructive_interference_size) std::atomic<uint32_t> _state{ 0 };
    };

    struct recursive_ticket_lock
    {
        struct recursive_ticket_lock_suspension
        {
            constexpr recursive_ticket_lock_suspension(recursive_ticket_lock& lock, uint32_t owner, uint32_t recursion) noexcept :
                _lock{ lock },
                _owner{ owner },
                _recursion{ recursion }
            {
            }

            // When this class is destroyed it restores the recursive_ticket_lock state.
            // This of course only works if the lock wasn't moved to another thread or something.
            recursive_ticket_lock_suspension(const recursive_ticket_lock_suspension&) = delete;
            recursive_ticket_lock_suspension& operator=(const recursive_ticket_lock_suspension&) = delete;
            recursive_ticket_lock_suspension(recursive_ticket_lock_suspension&&) = delete;
            recursive_ticket_lock_suspension& operator=(recursive_ticket_lock_suspension&&) = delete;

            ~recursive_ticket_lock_suspension()
            {
                if (_owner)
                {
                    // If someone reacquired the lock on the current thread, we shouldn't lock it again.
                    if (_lock._owner.load(std::memory_order_relaxed) != _owner)
                    {
                        _lock._lock.lock(); // lock-lock-lock lol
                        _lock._owner.store(_owner, std::memory_order_relaxed);
                    }
                    // ...but we should restore the original recursion count.
                    _lock._recursion += _recursion;
                }
            }

        private:
            friend struct recursive_ticket_lock;

            recursive_ticket_lock& _lock;
            uint32_t _owner = 0;
            uint32_t _recursion = 0;
        };

        void lock() noexcept
        {
            const auto id = GetCurrentThreadId();

            if (_owner.load(std::memory_order_relaxed) != id)
            {
                _lock.lock();
                _owner.store(id, std::memory_order_relaxed);
            }

            _recursion++;
        }

        void unlock() noexcept
        {
            if (--_recursion == 0)
            {
                _owner.store(0, std::memory_order_relaxed);
                _lock.unlock();
            }
        }

        [[nodiscard]] recursive_ticket_lock_suspension suspend() noexcept
        {
            const auto id = GetCurrentThreadId();
            uint32_t owner = 0;
            uint32_t recursion = 0;

            if (_owner.load(std::memory_order_relaxed) == id)
            {
                owner = id;
                recursion = _recursion;
                _owner.store(0, std::memory_order_relaxed);
                _recursion = 0;
                _lock.unlock();
            }

            return { *this, owner, recursion };
        }

        uint32_t is_locked() const noexcept
        {
            const auto id = GetCurrentThreadId();
            return _owner.load(std::memory_order_relaxed) == id;
        }

        uint32_t recursion_depth() const noexcept
        {
            return is_locked() ? _recursion : 0;
        }

    private:
        ticket_lock _lock;
        alignas(std::hardware_destructive_interference_size) std::atomic<uint32_t> _owner = 0;
        uint32_t _recursion = 0;
    };

    using recursive_ticket_lock_suspension = recursive_ticket_lock::recursive_ticket_lock_suspension;
}
