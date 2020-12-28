#ifndef timer_h_
#define timer_h_

class ScopedTimer
{
public:
  ScopedTimer(double &result)
    : result(result)
    , temp(std::chrono::system_clock::now())
  {}

  ~ScopedTimer()
  {
    result +=
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - temp)
        .count() /
      1e6;
  }

private:
  double &                                           result;
  std::chrono::time_point<std::chrono::system_clock> temp;
};

#endif