#pragma once
#include <algorithm>
#include <future>
#include <queue>
#include <type_traits>

// Simple ThreadPool implementation using C++20 jthread and tile-based tasks
class ThreadPool
{
  public:
	explicit ThreadPool(size_t threadCount)
	{
		for (size_t i = 0; i < threadCount; ++i)
		{
			workers.emplace_back([this](std::stop_token st)
			{
				std::unique_lock<std::mutex> lock(queueMutex, std::defer_lock);
				while (true)
				{
					std::function<void()> task;
					{
						std::lock_guard<std::mutex> guard(queueMutex);
						if (tasks.empty() && st.stop_requested())
							return;
					}
					lock.lock();
					condition.wait(lock, [&]
					{
						return st.stop_requested() || !tasks.empty();
					});
					if (st.stop_requested() && tasks.empty())
					{
						lock.unlock();
						return;
					}
					task = std::move(tasks.front());
					tasks.pop();
					lock.unlock();
					task();
				}
			});
		}
	}

	template<typename F, typename... Args>
	auto Enqueue(F&& f, Args&&... args)
	{
		using ReturnType = std::invoke_result_t<F, Args...>;
		auto taskPtr	 = std::make_shared<std::packaged_task<ReturnType()>>(
			[func = std::forward<F>(f), ... capturedArgs = std::forward<Args>(args)]() mutable
		{
			return std::invoke(func, capturedArgs...);
		});
		std::future<ReturnType> future = taskPtr->get_future();
		{
			std::lock_guard<std::mutex> lock(queueMutex);
			tasks.emplace([taskPtr]()
			{ (*taskPtr)(); });
		}
		condition.notify_one();
		return future;
	}

	~ThreadPool()
	{
		// Request all threads to stop
		for (auto& worker : workers)
			worker.request_stop();
		condition.notify_all();
		// jthread destructor joins
	}

  private:
	std::vector<std::jthread>		  workers;
	std::queue<std::function<void()>> tasks;
	std::mutex						  queueMutex;
	std::condition_variable			  condition;
};