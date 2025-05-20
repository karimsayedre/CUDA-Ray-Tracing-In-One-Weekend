#pragma once
#include <algorithm>
#include <future>
#include <queue>
#include <type_traits>

// Simple ThreadPool implementation using C++20 jthread and tile-based m_Tasks
class ThreadPool
{
  public:
	explicit ThreadPool()
	{
		for (size_t i = 0; i < std::thread::hardware_concurrency(); ++i)
		{
			m_Workers.emplace_back([this](const std::stop_token& st)
			{
				std::unique_lock<std::mutex> lock(m_QueueMutex, std::defer_lock);
				while (true)
				{
					std::function<void()> task;
					{
						std::lock_guard<std::mutex> guard(m_QueueMutex);
						if (m_Tasks.empty() && st.stop_requested())
							return;
					}
					lock.lock();
					m_Condition.wait(lock, [&]
					{
						return st.stop_requested() || !m_Tasks.empty();
					});
					if (st.stop_requested() && m_Tasks.empty())
					{
						lock.unlock();
						return;
					}
					task = std::move(m_Tasks.front());
					m_Tasks.pop();
					lock.unlock();
					task();
				}
			});
		}
	}

	template<typename F, typename... Args>
	[[nodiscard]] auto Enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
		requires std::invocable<F, Args...> && std::same_as<std::invoke_result_t<F, Args...>, void>
	{
		using ReturnType = std::invoke_result_t<F, Args...>;
		auto taskPtr	 = std::make_shared<std::packaged_task<ReturnType()>>(
			[func = std::forward<F>(f), ... capturedArgs = std::forward<Args>(args)]() mutable
		{
			return std::invoke(func, capturedArgs...);
		});
		std::future<ReturnType> future = taskPtr->get_future();
		{
			std::lock_guard<std::mutex> lock(m_QueueMutex);
			m_Tasks.emplace([taskPtr]()
			{ (*taskPtr)(); });
		}
		m_Condition.notify_one();
		return future;
	}

	void StopAndClear()
	{
		for (auto& worker : m_Workers)
			worker.request_stop();
		{
			std::lock_guard<std::mutex>		  lock(m_QueueMutex);
			std::queue<std::function<void()>> empty;
			std::swap(m_Tasks, empty);
		}
		m_Condition.notify_all();
	}

	~ThreadPool()
	{
		StopAndClear();
		// jthread destructor joins
	}

  private:
	std::vector<std::jthread>		  m_Workers;
	std::queue<std::function<void()>> m_Tasks;
	std::mutex						  m_QueueMutex;
	std::condition_variable			  m_Condition;
};