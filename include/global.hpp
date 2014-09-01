#include <stdint.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>

//using namespace boost;
using boost::shared_ptr;

typedef Eigen::Matrix<uint32_t,Eigen::Dynamic,1> VectorXu;
typedef Eigen::Matrix<uint32_t,Eigen::Dynamic,Eigen::Dynamic> MatrixXu;

// shared pointer typedefs
typedef boost::shared_ptr<Eigen::MatrixXd> spMatrixXd;
typedef boost::shared_ptr<Eigen::MatrixXf> spMatrixXf;

typedef boost::shared_ptr<Eigen::VectorXd> spVectorXd;
typedef boost::shared_ptr<Eigen::VectorXf> spVectorXf;
typedef boost::shared_ptr<Eigen::VectorXi> spVectorXi;
typedef boost::shared_ptr<VectorXu> spVectorXu;

#ifndef NDEBUG
#   define ASSERT(condition, message) \
  do { \
    if (! (condition)) { \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
      << " line " << __LINE__ << ": " << message << " " << std::endl; \
      assert(false); \
      std::exit(EXIT_FAILURE); \
    } \
  } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif
