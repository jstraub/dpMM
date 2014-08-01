#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <iostream>
#include <sys/time.h>
#include <string>

using namespace std;

class Timer
{
public:
  Timer()
  {
    gettimeofday(&tinit, NULL);
    tic();
  };

  void tic(void)
  {
    gettimeofday(&t0, NULL);
  };

  float toc(void)
  {
    gettimeofday(&t1, NULL);
    return getDtMs(t0);
  };

  float toctic(string description)
  {
    gettimeofday(&t1, NULL);
    float dt = getDtMs(t0);
    if (description.size()>0)
      cerr<<description<<": "<<dt<<"ms"<<endl;
    tic();
    return dt;
  };

  float lastDt(void) const
  {
    return dt;
  };

  float dtFromInit(void) 
  {
    return getDtMs(tinit);
  };

  Timer& operator=(const Timer& t)
  {
    if(this != &t){
      dt=t.lastDt();
    }
    return *this;
  };

private:
  float dt;
  timeval tinit, t0, t1;

  float getDtMs(const timeval& t) 
  {
    dt = (t1.tv_sec - t.tv_sec) * 1000.0f; // sec to ms
    dt += (t1.tv_usec - t.tv_usec) / 1000.0f; // us to ms
    return dt;
  };
};

inline ostream& operator<<(ostream &out, const Timer& t)
{
  out << t.lastDt() << "ms";
  return out;
};

#endif /* TIMER_HPP_ */
