#ifndef TIMER_HPP_
#define TIMER_HPP_

#ifdef WIN32   // Windows system specific
	#include <windows.h>
#else          // Unix based system specific
	#include <sys/time.h>
#endif

#include <iostream>
#include <string>

using namespace std;

#ifdef WIN32
	
	class Timer
	{
		public:
			//cross platform timer (only being used in windows)
			Timer()
			{
				#ifdef WIN32
					QueryPerformanceFrequency(&frequency);
					startCount.QuadPart = 0;
					endCount.QuadPart = 0;
					initCount.QuadPart = 0; 
				#else
					startCount.tv_sec = startCount.tv_usec = 0;
					endCount.tv_sec = endCount.tv_usec = 0;
					initCount.tv_sec = initCount.tv_usec = 0;
				#endif

				stopped = 0;
				startTimeInMicroSec = 0;
				endTimeInMicroSec = 0;

				startInit(); 
				tic();
			}                                    

			void tic()
			{
				this->start(); 
			};

			float toc()
			{
				this->stop();
				return float(getElapsedTimeInMilliSec());
			};

			float toctic(string description)
			{
				float dt = toc();
				if (description.size()>0)
					cerr<<description<<": "<<dt<<"ms"<<endl;
				tic();
				return dt;
			};

			float lastDt() const
			{
				return dt;
			}

			float dtFromInit() 
			{
				double tEnd = getEndTimeInMilliSec();
				double tStart = getElapsedTimeInMicroSec()*0.001;
				float dt = float(tEnd-tStart); 
				return dt;
			};

			Timer& operator=(const Timer& t)
			{
				if(this != &t){
					dt=t.lastDt();
				}
				return *this;
			};

			void startInit() {
				#ifdef WIN32
					QueryPerformanceCounter(&initCount);
				#else
					gettimeofday(&initCount, NULL);
				#endif
			} // start initialization timer

			void start() {
				stopped = 0; // reset stop flag
				#ifdef WIN32
					QueryPerformanceCounter(&startCount);
				#else
					gettimeofday(&startCount, NULL);
				#endif
			} // start timer

			void   stop() {
				stopped = 1; // set timer stopped flag

				#ifdef WIN32
					QueryPerformanceCounter(&endCount);
				#else
					gettimeofday(&endCount, NULL);
				#endif
			}                              // stop the timer

			double getElapsedTime(){
				return this->getElapsedTimeInSec();
			}                    // get elapsed time in second

			double getElapsedTimeInSec() {
				return this->getElapsedTimeInMicroSec() * 0.000001;
			}// get elapsed time in second (same as getElapsedTime)

			double getElapsedTimeInMilliSec() {
				return this->getElapsedTimeInMicroSec() * 0.001;
			}          // get elapsed time in milli-second

			double getElapsedTimeInMicroSec() {
				#ifdef WIN32
					if(!stopped)
						QueryPerformanceCounter(&endCount);

					startTimeInMicroSec = startCount.QuadPart * (1000000.0 / frequency.QuadPart);
					endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);
				#else
					if(!stopped)
						gettimeofday(&endCount, NULL);

					startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
					endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
				#endif
				return endTimeInMicroSec - startTimeInMicroSec;
			}          // get elapsed time in micro-second

	
			double getElapsedTimeInMicroSecFromInit() {
				#ifdef WIN32
					if(!stopped)
						QueryPerformanceCounter(&endCount);

					double startTimeInMicroSecInit = initCount.QuadPart * (1000000.0 / frequency.QuadPart);
					endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);
				#else
					if(!stopped)
						gettimeofday(&endCount, NULL);

					double startTimeInMicroSecInit = (initCount.tv_sec * 1000000.0) + initCount.tv_usec;
					endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
				#endif
				return endTimeInMicroSec - startTimeInMicroSecInit;
			}          // get elapsed time in micro-second from initialization

			void displayElapsedTime() { 
				double time; 
				time=this->getElapsedTimeInSec();
				std::cout << "[Timer::displayElapsedTime] Elapsed time " << time << " (s)." << std::endl;	
			}
	
			void displayElapsedTimeInMilliSec() {
				double time; 
				time=this->getElapsedTimeInMicroSec() * 0.001;
				std::cout << "[Timer::displayElapsedTimeInMilliSec] Elapsed time " << time << " (ms)." << std::endl;
			}          
    
			void displayElapsedTimeInMicroSec() {
				double time; 
				time=this->getElapsedTimeInMicroSec();
				std::cout << "[Timer::displayElapsedTimeInMicroSec] Elapsed time " << time << " (us)." << std::endl;	
			}

			void displayElapsedTimeAuto() { 
				double time; 
				time=this->getElapsedTimeInMicroSec();
				if(time<1000) {
					std::cout << "[Timer::displayElapsedTimeAuto] Elapsed time " << time << " (us)." << std::endl;	
				} else if(time >=1000 && time<1000000){
					std::cout << "[Timer::displayElapsedTimeAuto] Elapsed time " << time *.001 << " (ms)." << std::endl;	
				}else if(time>=1000000 && time<60000000) {
					std::cout << "[Timer::displayElapsedTimeAuto] Elapsed time " << time * 0.000001 << " (s)." << std::endl;	
				} else {
					time  *= 0.000001; //time in seconds 
					int hr, minutes, secs; 
					hr = floor(time/3600); 
					minutes = floor( (int) (time/60) % 60); 
					secs  = (int) time % 60; 
					std::cout <<  "[Timer::displayElapsedTimeAuto] Elapsed time: "; 
					if(hr>0) 
						std::cout << setw(2) << setfill('0') << hr << "hr:";

					std::cout << setw(2) << setfill('0') << minutes << "min:";
					std::cout << setw(2) << setfill('0') << secs << "secs." << std::endl;	
				} 

			}

			double getStartTime() 
			{
				return startTimeInMicroSec;
			}
			double getEndTime()
			{
				return endTimeInMicroSec; 
			}

			double getStartTimeInMilliSec() 
			{
				return(startTimeInMicroSec*0.001);
			}
			
			double getEndTimeInMilliSec()
			{
				return(endTimeInMicroSec*0.001);
			}

			private:
				double startTimeInMicroSec;                 // starting time in micro-second
				double endTimeInMicroSec;                   // ending time in micro-second
				int    stopped;                             // stop flag 
				float dt;									// last computed time 
				#ifdef WIN32
					LARGE_INTEGER frequency;                    // ticks per second
					LARGE_INTEGER startCount;                   //
					LARGE_INTEGER endCount;                     //
					LARGE_INTEGER initCount; 
				#else
					timeval startCount;                         //
					timeval endCount;                           //
					timeval initCount; 
				#endif

				float getDtMs(Timer& t) 
				{
					double tEnd = getEndTimeInMilliSec();
					double tStart = t.getStartTimeInMilliSec();
					dt = float(tEnd-tStart); 
					return dt;
				};
		};



#else 
	//original UNIX only timer 
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

#endif 



	inline ostream& operator<<(ostream &out, const Timer& t)
	{
	  out << t.lastDt() << "ms";
	  return out;
	};


#endif /* TIMER_HPP_ */
